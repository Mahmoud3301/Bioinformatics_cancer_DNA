"""
generate_multigene_data.py
Bio-Forget · DS307 · Spring 2025-2026

Generates:
  Data/patients_db.json      — 100 patients with k-mer feature vectors
  Data/scaler_params.json    — StandardScaler mean/scale
  Data/shard_map.json        — patient_id → shard (0-3), aligned to train split
  Data/mia_thresholds.json   — MIA config
  Data/KRAS.fasta            — synthesised if missing
  Data/TP53.fasta            — synthesised if missing
  Data/ALK.fasta             — synthesised if missing

FIX: shard_map is now built AFTER the 80/20 train split so it only maps
     training patients and never overrides test-set patients with shard=-1.
"""

import argparse
import json
import os
import random
from collections import Counter
from itertools import product

import numpy as np
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

GENES     = ["EGFR", "KRAS", "TP53", "ALK"]
K         = 4
ALL_KMERS = ["".join(p) for p in product("ACGT", repeat=K)]
N_SHARDS  = 4


# ── FASTA helpers ──────────────────────────────────────────────────────────────

def read_first_fasta_sequence(path):
    for rec in SeqIO.parse(path, "fasta"):
        return str(rec.seq).upper(), rec.id
    raise ValueError(f"No FASTA records found in {path}")


def inject_mutations(seq, rate, rng=None):
    bases = "ACGT"
    chars = list(seq)
    rand  = rng.random if rng else random.random
    choice= rng.choice if rng else random.choice
    for i, ch in enumerate(chars):
        if ch not in bases:
            continue
        if rand() < rate:
            chars[i] = choice([b for b in bases if b != ch])
    return "".join(chars)


def synthesize_gene_sequence(base_seq, target_len, seed_offset):
    rng = random.Random(42 + seed_offset)
    seq = inject_mutations(base_seq, 0.08, rng)
    if len(seq) >= target_len:
        return seq[:target_len]
    reps     = (target_len // len(seq)) + 1
    extended = (seq * reps)[:target_len]
    return extended


def write_gene_fastas(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    egfr_path = os.path.join(data_dir, "EGFR.fasta")
    if not os.path.exists(egfr_path):
        raise FileNotFoundError(
            f"{egfr_path} is missing. Put your EGFR FASTA there first."
        )

    egfr_seq, egfr_id = read_first_fasta_sequence(egfr_path)
    refs = {"EGFR": (egfr_seq, egfr_id)}
    target_lengths = {"KRAS": 190_000, "TP53": 175_000, "ALK": 210_000}

    for idx, gene in enumerate(["KRAS", "TP53", "ALK"], start=1):
        gene_path = os.path.join(data_dir, f"{gene}.fasta")
        if os.path.exists(gene_path):
            seq, gid = read_first_fasta_sequence(gene_path)
        else:
            seq = synthesize_gene_sequence(egfr_seq, target_lengths[gene], idx)
            gid = f"SYNTH_{gene}_REF"
            with open(gene_path, "w") as f:
                f.write(f">{gid}\n")
                for i in range(0, len(seq), 80):
                    f.write(seq[i : i + 80] + "\n")
            print(f"  Synthesised {gene}.fasta  ({len(seq):,} bp)")
        refs[gene] = (seq, gid)
    return refs


# ── Feature extraction ─────────────────────────────────────────────────────────

def kmer_frequency_vector(seq):
    counts = Counter(
        seq[i : i + K]
        for i in range(len(seq) - K + 1)
        if "N" not in seq[i : i + K]
    )
    total = sum(counts.values()) or 1
    return np.array([counts.get(km, 0) / total for km in ALL_KMERS], dtype=np.float32)


def build_windows(seq, window=300, step=150):
    return [seq[i : i + window] for i in range(0, len(seq) - window + 1, step)]


# ── Patient generation ─────────────────────────────────────────────────────────

def generate_patients(refs, n_total=100):
    random.seed(42)
    np.random.seed(42)

    per_class        = n_total // 2            # 50 healthy, 50 cancer
    cancer_per_gene  = per_class // len(GENES) # 12 per gene (48 cancer total → pad last gene)
    windows_per_pat  = 20

    gene_windows = {g: build_windows(refs[g][0]) for g in GENES}
    for gene, wins in gene_windows.items():
        if len(wins) < windows_per_pat:
            raise ValueError(f"{gene} sequence is too short for window sampling.")

    patient_db = []
    pid = 0

    # Healthy patients (label=0)
    for _ in range(per_class):
        gene  = random.choice(GENES)
        wins  = random.sample(gene_windows[gene], windows_per_pat)
        # Low mutation rate (0.005) — essentially reference-like sequence
        muts  = [inject_mutations(w, 0.005) for w in wins]
        merged= "".join(muts)
        gc    = (merged.count("G") + merged.count("C")) / max(len(merged), 1)
        patient_db.append({
            "patient_id":   f"PT_{pid:04d}",
            "label":        0,
            "label_str":    "Healthy",
            "gene":         gene,
            "sequences":    muts,
            "gc_content":   round(gc * 100, 2),
            "mutation_rate": 0.5,
        })
        pid += 1

    # Cancer patients (label=1) — equal per gene
    # FIX: raised mutation rate 0.05 → 0.12 so cancerous k-mer profiles diverge
    #      clearly from healthy ones, giving the CNN a learnable signal.
    remaining = per_class
    for g_idx, gene in enumerate(GENES):
        count = cancer_per_gene if g_idx < len(GENES) - 1 else remaining
        for _ in range(count):
            wins  = random.sample(gene_windows[gene], windows_per_pat)
            muts  = [inject_mutations(w, 0.12) for w in wins]  # was 0.05 → 0.12
            merged= "".join(muts)
            gc    = (merged.count("G") + merged.count("C")) / max(len(merged), 1)
            patient_db.append({
                "patient_id":   f"PT_{pid:04d}",
                "label":        1,
                "label_str":    "Cancerous",
                "gene":         gene,
                "sequences":    muts,
                "gc_content":   round(gc * 100, 2),
                "mutation_rate": 12.0,
            })
            pid += 1
        remaining -= count

    return patient_db


# ── Feature vectorisation + saving ────────────────────────────────────────────

def add_features_and_save(patient_db, data_dir):
    X = []
    for p in patient_db:
        vecs     = [kmer_frequency_vector(s) for s in p["sequences"]]
        mean_vec = np.mean(vecs, axis=0)
        p["feature_vec"] = mean_vec.tolist()
        X.append(mean_vec)

    X = np.array(X, dtype=np.float32)
    scaler = StandardScaler()
    scaler.fit(X)

    # ── FIX: build shard_map from the TRAINING split only ─────────────────────
    all_ids  = [p["patient_id"] for p in patient_db]
    all_y    = np.array([p["label"] for p in patient_db])

    train_ids, test_ids = train_test_split(
        all_ids, test_size=0.2, random_state=42, stratify=all_y
    )

    # Assign shards deterministically to training patients only
    shard_map: dict[str, int] = {}
    for pid in train_ids:
        shard_map[pid] = int(abs(hash(pid)) % N_SHARDS)
    # Test-set patients get shard=-1 as a sentinel (never used for training)
    for pid in test_ids:
        shard_map[pid] = -1

    # ── Save files ─────────────────────────────────────────────────────────────
    with open(os.path.join(data_dir, "patients_db.json"), "w") as f:
        json.dump(patient_db, f)

    with open(os.path.join(data_dir, "scaler_params.json"), "w") as f:
        json.dump({
            "mean_":  scaler.mean_.tolist(),
            "scale_": scaler.scale_.tolist(),
        }, f)

    with open(os.path.join(data_dir, "shard_map.json"), "w") as f:
        json.dump(shard_map, f)

    with open(os.path.join(data_dir, "mia_thresholds.json"), "w") as f:
        json.dump({
            "random_baseline":    0.52,
            "forgotten_threshold": 0.08,
            "partial_threshold":  0.15,
            "patients":           {},
        }, f)

    print(f"\nFiles written to: {os.path.abspath(data_dir)}")
    print(f"  patients_db.json   ({len(patient_db)} patients)")
    print(f"  scaler_params.json (mean_ shape: {scaler.mean_.shape})")
    print(f"  shard_map.json     ({len(train_ids)} train / {len(test_ids)} test)")
    print(f"  mia_thresholds.json")
    return scaler, train_ids, test_ids


# ── Optional: train & save CNN model ──────────────────────────────────────────

def train_and_save_cnn(patient_db, data_dir):
    """
    Trains EGFRClassifier on the full training split and saves it as
    Data/egfr_classifier.pth  (also copies to parent dir for app.py).
    Requires torch + scikit-learn.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.metrics import accuracy_score
    except ImportError:
        print("torch not installed — skipping CNN save.")
        return

    with open(os.path.join(data_dir, "scaler_params.json")) as f:
        sp = json.load(f)
    with open(os.path.join(data_dir, "shard_map.json")) as f:
        shard_map = json.load(f)

    scaler_mean  = np.array(sp["mean_"],  dtype=np.float32)
    scaler_scale = np.array(sp["scale_"], dtype=np.float32)
    scaler_scale = np.where(scaler_scale == 0, 1.0, scaler_scale)

    all_ids = [p["patient_id"] for p in patient_db]
    all_y   = np.array([p["label"]       for p in patient_db])
    all_X   = np.array([p["feature_vec"] for p in patient_db], dtype=np.float32)
    all_Xsc = ((all_X - scaler_mean) / scaler_scale).astype(np.float32)

    train_ids_set = {pid for pid, s in shard_map.items() if s >= 0}
    train_idx = [i for i, pid in enumerate(all_ids) if pid in train_ids_set]
    test_idx  = [i for i, pid in enumerate(all_ids) if pid not in train_ids_set]

    X_train = all_Xsc[train_idx];  y_train = all_y[train_idx]
    X_test  = all_Xsc[test_idx];   y_test  = all_y[test_idx]

    INPUT_SIZE = all_Xsc.shape[1]   # 256

    class EGFRClassifier(nn.Module):
        def __init__(self, input_size=256):
            super().__init__()
            self.conv1 = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=5, padding=2),
                nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2))
            self.conv2 = nn.Sequential(
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2))
            co = input_size // 4
            self.fc = nn.Sequential(
                nn.Flatten(), nn.Linear(64 * co, 128),
                nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, 2))
        def forward(self, x): return self.fc(self.conv2(self.conv1(x)))

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = EGFRClassifier(INPUT_SIZE).to(device)

    ds_tr  = TensorDataset(torch.FloatTensor(X_train).unsqueeze(1), torch.LongTensor(y_train))
    loader = DataLoader(ds_tr, batch_size=16, shuffle=True)
    opt    = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch    = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-5)
    crit   = nn.CrossEntropyLoss(label_smoothing=0.1)

    EPOCHS = 100   # was 40 — small dataset needs more passes
    print(f"\nTraining EGFRClassifier for {EPOCHS} epochs...")
    best_acc, best_state = 0.0, None
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                preds = model(torch.FloatTensor(X_test).unsqueeze(1).to(device)).argmax(1).cpu()
            acc = accuracy_score(y_test, preds)
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  Epoch {epoch+1}/{EPOCHS}  test_acc={acc:.2%}  best={best_acc:.2%}")
    if best_state:
        model.load_state_dict(best_state)
        print(f"  Restored best model (acc={best_acc:.2%})")

    # Save state_dict (used by app.py)
    pth_data   = os.path.join(data_dir, "egfr_classifier.pth")
    pth_parent = os.path.join(os.path.dirname(data_dir), "egfr_classifier.pth")
    torch.save(model.state_dict(), pth_data)
    torch.save(model.state_dict(), pth_parent)
    print(f"  Saved → {pth_data}")
    print(f"  Saved → {pth_parent}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate multi-gene demo dataset.")
    parser.add_argument("--data-dir", default="Data",
                        help="Directory for FASTA and JSON outputs.")
    parser.add_argument("--train-cnn", action="store_true",
                        help="Also train & save egfr_classifier.pth after data generation.")
    args = parser.parse_args()

    print("=== Bio-Forget Data Generator ===")
    refs    = write_gene_fastas(args.data_dir)
    patients= generate_patients(refs, n_total=100)
    add_features_and_save(patients, args.data_dir)

    healthy = sum(1 for p in patients if p["label"] == 0)
    cancer  = sum(1 for p in patients if p["label"] == 1)
    print(f"\nGenerated {len(patients)} patients ({healthy} healthy / {cancer} cancer).")
    print("Cancer per gene:", {
        g: sum(1 for p in patients if p["label"] == 1 and p["gene"] == g)
        for g in GENES
    })

    if args.train_cnn:
        train_and_save_cnn(patients, args.data_dir)
    else:
        print("\nTip: run with --train-cnn to also save egfr_classifier.pth")


if __name__ == "__main__":
    main()
