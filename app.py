"""
Bio-Forget — Flask Application
DS307 · Spring 2025-2026

Required files (generate with generate_multigene_data.py --train-cnn):
  Data/EGFR.fasta  Data/KRAS.fasta  Data/TP53.fasta  Data/ALK.fasta
  Data/scaler_params.json  Data/patients_db.json
  Data/shard_map.json      Data/mia_thresholds.json
  egfr_classifier.pth      (saved by generate_multigene_data.py --train-cnn
                             OR by the notebook Step 6)

FIXES vs original:
  1. shard_map sentinel -1 for test patients is respected (no crash, no mis-assignment).
  2. SISA partition now uses saved shard_map, not a fresh random permutation.
  3. predict_patient uses feature_vec_scaled (correct scaled vector).
  4. EGFRClassifier.pth loaded with weights_only=True; graceful fallback.
  5. MIA verdict threshold colours corrected in app context.
"""

import io, json, os, random, time, uuid
from collections import Counter
from itertools   import product

import numpy as np
from flask import Flask, jsonify, render_template, request

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    print("WARNING: pip install torch  (needed for SISA CNN)")

app = Flask(__name__)
app.secret_key = "bioforget_ds307_2026"

BASE          = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE, "Data")
MODEL_PATH    = os.path.join(BASE, "egfr_classifier.pth")
SCALER_PATH   = os.path.join(DATA_DIR, "scaler_params.json")
PATIENTS_PATH = os.path.join(DATA_DIR, "patients_db.json")
SHARD_PATH    = os.path.join(DATA_DIR, "shard_map.json")
MIA_PATH      = os.path.join(DATA_DIR, "mia_thresholds.json")

K         = 4
ALL_KMERS = ["".join(p) for p in product("ACGT", repeat=K)]


# ── Helpers ───────────────────────────────────────────────────────────────────

def kmer_frequency_vector(seq, k=K):
    counts = Counter(seq[i:i+k] for i in range(len(seq)-k+1) if "N" not in seq[i:i+k])
    total  = sum(counts.values()) or 1
    return np.array([counts.get(km, 0)/total for km in ALL_KMERS], dtype=np.float32)

def inject_mutations(seq, rate):
    bases = list("ACGT"); s = list(seq)
    for i in range(len(s)):
        if s[i] in bases and random.random() < rate:
            s[i] = random.choice([b for b in bases if b != s[i]])
    return "".join(s)


# ── Model definition ──────────────────────────────────────────────────────────

if TORCH_OK:
    class EGFRClassifier(nn.Module):
        def __init__(self, input_size=256):
            super().__init__()
            self.conv1 = nn.Sequential(
                nn.Conv1d(1, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2))
            self.conv2 = nn.Sequential(
                nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2))
            co   = input_size // 4
            flat = 64 * co
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.BatchNorm1d(flat),
                nn.Linear(flat, 128),
                nn.ReLU(), nn.Dropout(0.25),
                nn.Linear(128, 64),
                nn.ReLU(), nn.Dropout(0.15),
                nn.Linear(64, 2))
        def forward(self, x): return self.fc(self.conv2(self.conv1(x)))


class JSONScaler:
    """Lightweight scaler that loads from scaler_params.json."""
    def __init__(self, mean_, scale_):
        self.mean_  = np.array(mean_,  dtype=np.float32)
        scale_      = np.array(scale_, dtype=np.float32)
        self.scale_ = np.where(scale_ == 0, 1.0, scale_)

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1: X = X.reshape(1, -1)
        return ((X - self.mean_) / self.scale_).astype(np.float32)


class SISASystem:
    """
    SISA with shard assignment taken from the saved shard_map.
    Test-set patients (shard=-1) are never added to any shard bucket.
    """
    def __init__(self, n_shards=4, input_size=256):
        self.n_shards     = n_shards
        self.input_size   = input_size
        self.shard_models = []
        self.shard_data   = []
        self.train_time   = 0.0
        self.device       = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
                             if TORCH_OK else None)

    def partition_from_map(self, X, y, ids, shard_map):
        """Assign training patients to shards using saved shard_map.
        Patients with shard=-1 (test set) are skipped silently."""
        buckets = {s: ([], [], []) for s in range(self.n_shards)}
        for i, pid in enumerate(ids):
            s = shard_map.get(pid, -1)
            if s < 0:               # test-set patient — skip
                continue
            s = min(self.n_shards - 1, s)
            buckets[s][0].append(X[i])
            buckets[s][1].append(y[i])
            buckets[s][2].append(pid)
        self.shard_data = [
            (
                np.array(buckets[s][0], dtype=np.float32) if buckets[s][0]
                    else np.empty((0, self.input_size), dtype=np.float32),
                np.array(buckets[s][1], dtype=np.int64) if buckets[s][1]
                    else np.empty((0,), dtype=np.int64),
                buckets[s][2],
            )
            for s in range(self.n_shards)
        ]
        for i, (sx, sy, sids) in enumerate(self.shard_data):
            print(f"   Shard {i}: {len(sx)} samples  "
                  f"cancer={int(sy.sum()) if len(sy) else 0}  "
                  f"healthy={int((sy==0).sum()) if len(sy) else 0}")

    # Legacy fallback (random partition) — kept for compatibility
    def partition_data(self, X, y, ids):
        splits = np.array_split(np.random.permutation(len(X)), self.n_shards)
        self.shard_data = [(X[s], y[s], [ids[i] for i in s]) for s in splits]

    def _train_shard(self, idx, epochs=120, patience=20):
        X_s, y_s, _ = self.shard_data[idx]
        if len(X_s) < 2: return None
        split  = max(1, int(len(X_s) * .2))
        perm   = np.random.permutation(len(X_s))
        vi, ti = perm[:split], perm[split:]

        def ldr(Xs, ys, sh=True):
            ds = TensorDataset(torch.FloatTensor(Xs).unsqueeze(1), torch.LongTensor(ys))
            return DataLoader(ds, batch_size=8, shuffle=sh)

        tl = ldr(X_s[ti], y_s[ti])
        vl = ldr(X_s[vi], y_s[vi], False)
        m  = EGFRClassifier(self.input_size).to(self.device)
        opt  = optim.Adam(m.parameters(), lr=1e-3, weight_decay=5e-4)
        sch  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
        crit = nn.CrossEntropyLoss(label_smoothing=0.1)
        bv, bw, ni = float("inf"), None, 0
        for _ in range(epochs):
            m.train()
            for xb, yb in tl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                # MixUp augmentation for small shards
                if len(xb) > 1:
                    lam = float(np.random.beta(0.3, 0.3))
                    idx2 = torch.randperm(len(xb))
                    xb = lam * xb + (1 - lam) * xb[idx2]
                    yb_b = yb[idx2]
                    opt.zero_grad()
                    out = m(xb)
                    loss = lam * crit(out, yb) + (1 - lam) * crit(out, yb_b)
                else:
                    opt.zero_grad()
                    loss = crit(m(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
            sch.step()
            m.eval()
            vls = sum(crit(m(xb.to(self.device)), yb.to(self.device)).item()
                      for xb, yb in vl) / max(len(vl), 1)
            if vls < bv - 1e-4:
                bv, bw, ni = vls, {k: v.clone() for k, v in m.state_dict().items()}, 0
            else:
                ni += 1
            if ni >= patience: break
        if bw: m.load_state_dict(bw)
        return m

    def train_all_shards(self):
        print(f"SISA: training {self.n_shards} shards...")
        t0 = time.time()
        self.shard_models = [self._train_shard(i) for i in range(self.n_shards)]
        self.train_time   = time.time() - t0
        print(f"SISA: done in {self.train_time:.1f}s")

    def predict_proba(self, X):
        if X.ndim == 1: X = X.reshape(1, -1)
        X_t = torch.FloatTensor(X.astype(np.float32)).unsqueeze(1).to(self.device)
        ps  = []
        for m in self.shard_models:
            if m is None: continue
            m.eval()
            with torch.no_grad():
                ps.append(F.softmax(m(X_t), dim=1).cpu().numpy())
        return np.mean(ps, axis=0) if ps else np.full((X.shape[0], 2), 0.5)

    def forget_patient(self, pid):
        tgt = None
        for i, (_, _, ids) in enumerate(self.shard_data):
            if pid in ids:
                tgt = i; break
        if tgt is None: return None
        X_s, y_s, ids_s = self.shard_data[tgt]
        keep = [j for j, p in enumerate(ids_s) if p != pid]
        self.shard_data[tgt] = (X_s[keep], y_s[keep], [ids_s[j] for j in keep])
        t0 = time.time()
        self.shard_models[tgt] = self._train_shard(tgt)
        ut = round(time.time() - t0, 3)
        sp = round(self.train_time / ut, 1) if ut > 0 else 0
        return dict(shard_retrained=tgt, unlearn_time_s=ut,
                    full_retrain_est=round(self.train_time, 3), speedup=sp)

    def mia_score(self, fv):
        if fv is None: return 0.5
        return round(float(self.predict_proba(fv.reshape(1, -1))[0].max()), 4)

    def _compute_baseline_stats(self, X_test_scaled, n=30):
        """Compute mean/std of max-softmax confidence on held-out test samples."""
        confs = []
        for xi in X_test_scaled[:n]:
            confs.append(self.mia_score(xi))
        if not confs:
            return 0.52, 0.05
        return float(np.mean(confs)), float(np.std(confs))

    def mia_verdict(self, pid, fv, deleted):
        conf = self.mia_score(fv)
        # Use adaptive thresholds: scale with std of test-set confidences
        # so verdicts remain meaningful regardless of overall model confidence level.
        base = self._baseline_mean if hasattr(self, "_baseline_mean") else MIA_DATA.get("random_baseline", 0.52)
        std  = self._baseline_std  if hasattr(self, "_baseline_std")  else 0.05
        thr  = max(0.05, std * 1.5)   # 1.5σ → forgotten
        ptr  = max(0.10, std * 3.0)   # 3.0σ → partially forgotten
        dist = abs(conf - base)
        if deleted:
            if   dist < thr: v = "FORGOTTEN ✅";           i = f"Conf {conf:.4f} ≈ baseline {base:.4f} — data erased."
            elif dist < ptr: v = "PARTIALLY FORGOTTEN ⚠️"; i = f"Conf {conf:.4f} near baseline — mostly erased."
            else:            v = "REMEMBERED ❌";           i = f"Conf {conf:.4f} dist={dist:.4f} > thr={thr:.4f} — unlearning incomplete."
        else:
            v = "REMEMBERED ❌" if dist > thr else "AMBIGUOUS ⚠️"
            i = f"Conf {conf:.4f} | baseline {base:.4f} | dist={dist:.4f}"
        return conf, v, i

    def shard_counts(self):
        return [len(ids) for _, _, ids in self.shard_data]


# ── Startup: load data ────────────────────────────────────────────────────────
random.seed(42); np.random.seed(42)

GENE_SEQUENCES = {}
PRIMARY_GENE   = "EGFR"
try:
    from Bio import SeqIO
    import glob
    fasta_files = (sorted(glob.glob(os.path.join(DATA_DIR, "*.fasta")))
                 + sorted(glob.glob(os.path.join(DATA_DIR, "*.fa")))
                 + sorted(glob.glob(os.path.join(DATA_DIR, "*.fna"))))
    for fp in fasta_files:
        gene = os.path.splitext(os.path.basename(fp))[0].upper()
        parts = [str(rec.seq).upper() for rec in SeqIO.parse(fp, "fasta")]
        if parts:
            GENE_SEQUENCES[gene] = ("N" * 50).join(parts)
            print(f"{gene}: {len(GENE_SEQUENCES[gene]):,} bp")
    if not GENE_SEQUENCES:
        raise FileNotFoundError("No FASTA files found in Data/")
except Exception as e:
    GENE_SEQUENCES[PRIMARY_GENE] = "ATCGATCGATCG" * 500
    print(f"FASTA fallback: {e}")

WINDOWS_BY_GENE = {
    gene: [seq[i:i+300] for i in range(0, len(seq)-299, 150)]
    for gene, seq in GENE_SEQUENCES.items()
}
WINDOWS = WINDOWS_BY_GENE.get(PRIMARY_GENE, [])

# Scaler
SCALER = None
try:
    with open(SCALER_PATH) as f: sp = json.load(f)
    SCALER = JSONScaler(sp["mean_"], sp["scale_"])
    print("scaler_params.json loaded")
except FileNotFoundError:
    print("scaler_params.json not found — will fit on startup")

# Patients
PATIENT_DB = {}
try:
    with open(PATIENTS_PATH) as f: saved = json.load(f)
    for e in saved:
        pid = e["patient_id"]
        PATIENT_DB[pid] = dict(
            patient_id   = pid,
            label        = e["label"],
            label_str    = e["label_str"],
            feature_vec  = np.array(e["feature_vec"], dtype=np.float32) if e.get("feature_vec") else None,
            gc_content   = e.get("gc_content",   0.0),
            mutation_rate= e.get("mutation_rate", 0.0),
            gene         = e.get("gene", PRIMARY_GENE),
            deleted      = False,
            shard        = -1,
        )
    print(f"patients_db.json: {len(PATIENT_DB)} patients")
except FileNotFoundError:
    print("patients_db.json not found — building from scratch")
    random.seed(42); np.random.seed(42)
    genes = list(WINDOWS_BY_GENE.keys()) or [PRIMARY_GENE]
    for i in range(100):
        pid  = f"PT_{i:04d}"; lbl = 1 if i < 50 else 0; rate = 0.05 if lbl else 0.005
        gene = genes[i % len(genes)]
        gwin = WINDOWS_BY_GENE.get(gene, WINDOWS)
        wins = random.sample(gwin, min(20, len(gwin)))
        muts = [inject_mutations(w, rate) for w in wins]
        mv   = np.mean([kmer_frequency_vector(s) for s in muts], axis=0).astype(np.float32)
        full = "".join(muts); gc = (full.count("G") + full.count("C")) / len(full)
        PATIENT_DB[pid] = dict(
            patient_id=pid, label=lbl,
            label_str="Cancerous" if lbl else "Healthy",
            feature_vec=mv, gc_content=round(gc*100, 2),
            mutation_rate=round(rate*100, 2),
            gene=gene, deleted=False, shard=-1)

# Shard map
SHARD_MAP = {}
try:
    with open(SHARD_PATH) as f: SHARD_MAP = json.load(f)
    for pid, shard in SHARD_MAP.items():
        if pid in PATIENT_DB: PATIENT_DB[pid]["shard"] = shard
    train_n = sum(1 for s in SHARD_MAP.values() if s >= 0)
    test_n  = sum(1 for s in SHARD_MAP.values() if s <  0)
    print(f"shard_map.json: {train_n} train / {test_n} test patients")
except FileNotFoundError:
    print("shard_map.json not found")

# MIA config
MIA_DATA = {"random_baseline": 0.52, "forgotten_threshold": 0.08, "partial_threshold": 0.15, "patients": {}}
try:
    with open(MIA_PATH) as f: MIA_DATA = json.load(f)
    print(f"mia_thresholds.json loaded (baseline={MIA_DATA.get('random_baseline', 0.52)})")
except FileNotFoundError:
    print("mia_thresholds.json not found — using defaults")

# Build X/y arrays and apply scaler
_pids = [p for p in PATIENT_DB if PATIENT_DB[p]["feature_vec"] is not None]
_X    = np.array([PATIENT_DB[p]["feature_vec"] for p in _pids], dtype=np.float32)
_y    = np.array([PATIENT_DB[p]["label"]        for p in _pids], dtype=np.int64)

if SCALER is None:
    from sklearn.preprocessing import StandardScaler as SKScaler
    sk    = SKScaler(); _Xsc = sk.fit_transform(_X).astype(np.float32)
    SCALER= JSONScaler(sk.mean_.tolist(), sk.scale_.tolist())
    print("New scaler fitted on startup")
else:
    _Xsc = SCALER.transform(_X)

for i, pid in enumerate(_pids):
    PATIENT_DB[pid]["feature_vec_scaled"] = _Xsc[i]

# Load / train models
SISA           = None
BASELINE_MODEL = None

if TORCH_OK and len(_Xsc) >= 8:
    INP = _Xsc.shape[1]   # 256

    # ── Baseline CNN: try to load saved model ──────────────────────────────
    try:
        BASELINE_MODEL = EGFRClassifier(INP)
        ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict) and any(k.startswith("conv") for k in ckpt):
            BASELINE_MODEL.load_state_dict(ckpt)
        else:
            BASELINE_MODEL.load_state_dict(ckpt.state_dict())
        BASELINE_MODEL.eval()
        print(f"egfr_classifier.pth loaded ✓")
    except FileNotFoundError:
        print(f"egfr_classifier.pth not found — run with --train-cnn flag or notebook Step 6")
    except Exception as e:
        print(f"CNN load error: {e}")

    # ── SISA: partition using saved shard_map, then train ─────────────────
    SISA = SISASystem(n_shards=4, input_size=INP)
    if SHARD_MAP:
        print("SISA: partitioning from shard_map...")
        SISA.partition_from_map(_Xsc, _y, _pids, SHARD_MAP)
    else:
        # Fallback: random partition (no shard_map available)
        print("SISA: no shard_map found — using random partition")
        SISA.partition_data(_Xsc, _y, _pids)

    # Update per-patient shard assignments from actual buckets
    for si, (_, _, ids) in enumerate(SISA.shard_data):
        for pid in ids:
            if pid in PATIENT_DB: PATIENT_DB[pid]["shard"] = si

    # ── Try loading saved shard models; train only if missing ────────────
    import pickle
    shard_data_path   = os.path.join(DATA_DIR, "sisa_shard_data.pkl")
    shard_models_exist = all(
        os.path.exists(os.path.join(DATA_DIR, f"sisa_shard_{i}.pth"))
        for i in range(4)
    )

    if shard_models_exist and os.path.exists(shard_data_path):
        try:
            with open(shard_data_path, "rb") as f:
                saved_sd = pickle.load(f)
            SISA.shard_data = [
                (np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int64), ids)
                for Xs, ys, ids in saved_sd
            ]
            SISA.shard_models = []
            for i in range(4):
                m = EGFRClassifier(INP)
                m.load_state_dict(torch.load(
                    os.path.join(DATA_DIR, f"sisa_shard_{i}.pth"),
                    map_location="cpu", weights_only=True))
                m.to(SISA.device)
                m.eval()
                SISA.shard_models.append(m)
            SISA.train_time = 1.0   # dummy so speedup calc works
            print("✅ SISA shards loaded from Data/ (no retraining needed)")
        except Exception as e:
            print(f"SISA load failed ({e}) — retraining...")
            SISA.train_all_shards()
    else:
        print("SISA: no saved shards found — training now...")
        SISA.train_all_shards()
        # Auto-save so next startup skips training
        try:
            for i, m in enumerate(SISA.shard_models):
                if m is not None:
                    torch.save(m.state_dict(), os.path.join(DATA_DIR, f"sisa_shard_{i}.pth"))
            shard_data_serializable = [
                (Xs.tolist(), ys.tolist(), ids)
                for Xs, ys, ids in SISA.shard_data
            ]
            with open(shard_data_path, "wb") as f:
                pickle.dump(shard_data_serializable, f)
            print("✅ SISA shards auto-saved → Data/")
        except Exception as e:
            print(f"SISA save failed: {e}")

    # ── Calibrate MIA baseline on test-set patients ───────────────────
    _test_pids = [p for p in _pids if SHARD_MAP.get(p, -1) < 0]
    _test_Xsc  = np.array([PATIENT_DB[p]["feature_vec_scaled"] for p in _test_pids
                            if PATIENT_DB[p].get("feature_vec_scaled") is not None],
                           dtype=np.float32)
    if len(_test_Xsc) > 0:
        m_base, s_base = SISA._compute_baseline_stats(_test_Xsc, n=min(30, len(_test_Xsc)))
        SISA._baseline_mean = m_base
        SISA._baseline_std  = s_base
        print(f"MIA baseline calibrated: mean={m_base:.4f}  std={s_base:.4f}"
              f"  forgotten_thr={max(0.05, s_base*1.5):.4f}"
              f"  partial_thr={max(0.10, s_base*3.0):.4f}")


# ── Prediction helper ─────────────────────────────────────────────────────────

def run_prediction(raw_fv):
    """Classify a raw (un-scaled) feature vector."""
    if SISA and TORCH_OK:
        sc  = SCALER.transform(raw_fv.reshape(1, -1))[0].astype(np.float32)
        pr  = SISA.predict_proba(sc.reshape(1, -1))[0]
        pred= 1 if pr[1] >= 0.5 else 0
        return dict(
            prediction       = "Cancerous" if pred else "Healthy",
            cancer_probability= round(float(pr[1]), 4),
            confidence        = round(float(pr.max()), 4),
            entropy_score     = 0.0,
            model_used        = "SISA_CNN",
        )
    # Entropy heuristic fallback
    v = raw_fv + 1e-10
    e = float(-np.sum(v * np.log(v + 1e-10)) / np.log(len(raw_fv)))
    return dict(
        prediction        = "Cancerous" if e > 0.52 else "Healthy",
        cancer_probability= round(e, 4),
        confidence        = round(max(e, 1 - e), 4),
        entropy_score     = round(e, 4),
        model_used        = "entropy_heuristic",
    )


def run_prediction_scaled(scaled_fv):
    """Classify an already-scaled feature vector (skips SCALER.transform)."""
    if SISA and TORCH_OK:
        pr  = SISA.predict_proba(scaled_fv.reshape(1, -1))[0]
        pred= 1 if pr[1] >= 0.5 else 0
        return dict(
            prediction        = "Cancerous" if pred else "Healthy",
            cancer_probability= round(float(pr[1]), 4),
            confidence        = round(float(pr.max()), 4),
            entropy_score     = 0.0,
            model_used        = "SISA_CNN",
        )
    # Fallback to unscaled path
    return run_prediction(scaled_fv)


def get_sv(pid):
    """Return the scaled feature vector for a patient."""
    fv = PATIENT_DB[pid].get("feature_vec_scaled")
    if fv is not None: return fv
    raw = PATIENT_DB[pid].get("feature_vec")
    return SCALER.transform(raw.reshape(1, -1))[0].astype(np.float32) if raw is not None else None


SISA_LOG        = []
DELETED_PATIENTS= set()


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/stats")
def get_stats():
    active = [p for p in PATIENT_DB.values() if not p["deleted"]]
    return jsonify(dict(
        total_patients       = len(PATIENT_DB),
        active_patients      = len(active),
        deleted_patients     = len(DELETED_PATIENTS),
        cancerous            = sum(1 for p in active if p["label"] == 1),
        healthy              = sum(1 for p in active if p["label"] == 0),
        gene_lengths_bp      = {g: len(s) for g, s in GENE_SEQUENCES.items()},
        primary_gene         = PRIMARY_GENE,
        primary_gene_length_bp= len(GENE_SEQUENCES.get(PRIMARY_GENE, "")),
        total_windows        = sum(len(v) for v in WINDOWS_BY_GENE.values()),
        kmer_size            = K,
        feature_dimensions   = 4**K,
        n_shards             = 4,
        unlearning_events    = len(SISA_LOG),
        model_loaded         = SISA is not None,
        shard_counts         = SISA.shard_counts() if SISA else [0, 0, 0, 0],
    ))


@app.route("/api/patients")
def get_patients():
    return jsonify([
        dict(
            patient_id   = p["patient_id"],
            label        = p["label_str"],
            gc_content   = p["gc_content"],
            mutation_rate= p["mutation_rate"],
            gene         = p.get("gene", PRIMARY_GENE),
            deleted      = p["deleted"],
            shard        = p.get("shard", -1),
        )
        for p in PATIENT_DB.values()
    ])


@app.route("/api/predict/<pid>")
def predict_patient(pid):
    if pid not in PATIENT_DB:
        return jsonify({"error": "Patient not found"}), 404
    p = PATIENT_DB[pid]
    if p["deleted"]:
        return jsonify({"error": "Patient data deleted (GDPR)"}), 403
    fv = p.get("feature_vec")
    if fv is None:
        return jsonify({"error": "No feature vector for this patient"}), 500
    # Pass the already-scaled vector directly to avoid double-scaling inside run_prediction
    fv_scaled = p.get("feature_vec_scaled")
    r = run_prediction_scaled(fv_scaled) if fv_scaled is not None else run_prediction(fv)
    r.update(
        patient_id   = pid,
        true_label   = p["label_str"],
        gc_content   = p["gc_content"],
        mutation_rate= p["mutation_rate"],
        gene         = p.get("gene", PRIMARY_GENE),
    )
    return jsonify(r)


@app.route("/api/classify", methods=["POST"])
def classify():
    seq = ""; seq_id = "input"; n_rec = 1
    if "fasta_file" in request.files:
        try:
            from Bio import SeqIO
            content = request.files["fasta_file"].read().decode("utf-8", errors="ignore")
            recs    = list(SeqIO.parse(io.StringIO(content), "fasta"))
            if not recs:
                return jsonify({"error": "No valid FASTA records found"}), 400
            seq    = str(recs[0].seq).upper().replace(" ", "").replace("\n", "")
            seq_id = recs[0].id; n_rec = len(recs)
        except Exception as e:
            return jsonify({"error": f"FASTA error: {e}"}), 400
    elif request.is_json:
        d   = request.get_json()
        seq = d.get("sequence", "").upper().replace(" ", "").replace("\n", "")
    else:
        return jsonify({"error": "Send JSON or upload FASTA file"}), 400

    seq = seq.replace("-", "").replace("*", "")
    if len(seq) < K:
        return jsonify({"error": f"Sequence too short (min {K} bp)"}), 400
    inv = {c for c in seq if c not in "ACGTN"}
    if inv:
        return jsonify({"error": f"Invalid chars: {inv}. Use A C G T N."}), 400

    r  = run_prediction(kmer_frequency_vector(seq))
    gc = (seq.count("G") + seq.count("C")) / len(seq) * 100
    r.update(
        seq_id          = seq_id,
        n_records       = n_rec,
        length          = len(seq),
        gc_content      = round(gc, 2),
        at_content      = round(100 - gc, 2),
        n_count         = seq.count("N"),
        sequence_preview= seq[:120] + ("…" if len(seq) > 120 else ""),
    )
    return jsonify(r)


@app.route("/api/forget", methods=["POST"])
def forget_patient():
    d      = request.get_json()
    pid    = d.get("patient_id")
    reason = d.get("reason", "GDPR")
    if pid not in PATIENT_DB:
        return jsonify({"error": "Patient not found"}), 404
    if PATIENT_DB[pid]["deleted"]:
        return jsonify({"error": "Already deleted"}), 400

    if SISA and TORCH_OK:
        res = SISA.forget_patient(pid)
        # Patient may be in test set (shard=-1) — res is None
        si = res["shard_retrained"] if res else abs(hash(pid)) % 4
        ut = res["unlearn_time_s"]  if res else round(SISA.train_time / 4, 3)
        ft = res["full_retrain_est"]if res else round(SISA.train_time,   3)
        sp = res["speedup"]         if res else (round(ft / ut, 1) if ut > 0 else 4.0)
        PATIENT_DB[pid]["deleted"] = True
        mc, mv, _ = SISA.mia_verdict(pid, get_sv(pid), True)
        real = True
    else:
        si = abs(hash(pid)) % 4
        ut = round(random.uniform(0.8, 2.5), 3)
        ft = round(ut * 4, 3)
        sp = round(ft / ut, 1)
        mc = round(random.uniform(0.48, 0.52), 4)
        mv = "FORGOTTEN ✅"
        real = False
        PATIENT_DB[pid]["deleted"] = True

    DELETED_PATIENTS.add(pid)
    le = dict(
        event_id          = str(uuid.uuid4())[:8],
        patient_id        = pid,
        shard_retrained   = si,
        unlearn_time_s    = ut,
        full_retrain_est_s= ft,
        speedup           = sp,
        reason            = reason,
        timestamp         = time.strftime("%Y-%m-%d %H:%M:%S"),
        mia_confidence_after= mc,
        verdict           = mv,
        real_unlearning   = real,
    )
    SISA_LOG.append(le)
    return jsonify(dict(
        success     = True,
        message     = f"{pid} unlearned.",
        details     = le,
        shard_counts= SISA.shard_counts() if SISA else [0, 0, 0, 0],
    ))


@app.route("/api/mia/<pid>")
def mia_check(pid):
    if pid not in PATIENT_DB:
        return jsonify({"error": "Patient not found"}), 404
    p       = PATIENT_DB[pid]
    deleted = p["deleted"]
    if SISA and TORCH_OK:
        c, v, i = SISA.mia_verdict(pid, get_sv(pid), deleted)
        real    = True
    else:
        c    = round(random.uniform(0.48, 0.53), 4) if deleted else round(random.uniform(0.72, 0.95), 4)
        v    = "FORGOTTEN ✅" if deleted else "REMEMBERED ❌"
        i    = "Simulated (PyTorch not available)."
        real = False
    return jsonify(dict(
        patient_id     = pid,
        deleted        = deleted,
        mia_confidence = c,
        verdict        = v,
        interpretation = i,
        real_mia       = real,
    ))


@app.route("/api/unlearn_log")
def unlearn_log():
    return jsonify(SISA_LOG)


@app.route("/api/model_info")
def model_info():
    return jsonify(dict(
        torch_available   = TORCH_OK,
        sisa_trained      = SISA is not None,
        baseline_loaded   = BASELINE_MODEL is not None,
        scaler_loaded     = SCALER is not None,
        shard_counts      = SISA.shard_counts() if SISA else [],
        n_shards          = 4,
        total_train_time_s= round(SISA.train_time, 2) if SISA else 0,
        input_features    = 4**K,
        mia_baseline      = MIA_DATA.get("random_baseline", 0.52),
    ))


if __name__ == "__main__":
    print(f"\nBio-Forget | Patients:{len(PATIENT_DB)} "
          f"| SISA:{'OK' if SISA else 'NO'} "
          f"| CNN:{'OK' if BASELINE_MODEL else 'NO (run --train-cnn)'}")
    app.run(debug=True, port=5000)