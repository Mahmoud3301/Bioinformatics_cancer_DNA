/* ═══════════════════════════════════════════════════════════
   Bio-Forget — Frontend Logic
   DS307 · Spring 2026
   ═══════════════════════════════════════════════════════════ */

let allPatients  = [];
let maxShardSize = 1;     // for bar width scaling
let sisTrained   = false; // whether SISA CNN is live

// ── view navigation ─────────────────────────────────────────────
function showView(name) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(`view-${name}`).classList.add('active');
  document.querySelector(`[data-view="${name}"]`).classList.add('active');
  if (name === 'patients') loadPatients();
  if (name === 'audit')    loadAuditLog();
  if (name === 'unlearn' || name === 'analyze') populateSelects();
}

// ── model info + status pill ────────────────────────────────────
async function loadModelInfo() {
  try {
    const d = await fetch('/api/model_info').then(r => r.json());
    sisTrained = d.sisa_trained;

    const pill = document.getElementById('model-status-pill');
    if (d.sisa_trained) {
      pill.className = 'model-pill model-pill-ok';
      pill.innerHTML = '<i class="fa-solid fa-circle-check"></i><span>SISA CNN Live</span>';
    } else {
      pill.className = 'model-pill model-pill-warn';
      pill.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i><span>Heuristic Mode</span>';
    }

    // Dashboard banner
    const banner = document.getElementById('model-banner');
    banner.classList.remove('hidden');
    if (d.sisa_trained) {
      banner.className = 'model-banner model-banner-ok';
      document.getElementById('banner-icon').className  = 'fa-solid fa-brain';
      document.getElementById('banner-title').textContent =
        `Real SISA CNN active — ${d.input_features}-feature k-mer vectors`;
      document.getElementById('banner-sub').textContent =
        `${d.n_shards || 4} shards trained in ${d.total_train_time_s}s  ·  `+
        `Baseline CNN ${d.baseline_loaded ? 'loaded ✓' : 'not found (SISA used)'}  ·  `+
        `Scaler ${d.scaler_loaded ? 'loaded ✓' : 'fitted on startup'}`;
    } else {
      banner.className = 'model-banner model-banner-warn';
      document.getElementById('banner-icon').className  = 'fa-solid fa-triangle-exclamation';
      document.getElementById('banner-title').textContent = 'PyTorch not available — using entropy heuristic';
      document.getElementById('banner-sub').textContent  = 'Install PyTorch to enable real CNN predictions and SISA unlearning.';
    }

    // Analyze tag
    const tag = document.getElementById('analyze-model-tag');
    if (tag) tag.textContent = d.sisa_trained ? 'Powered by SISA CNN' : 'Entropy heuristic mode';

    // Shard bars
    if (d.shard_counts && d.shard_counts.length) {
      maxShardSize = Math.max(...d.shard_counts, 1);
      updateShardBars(d.shard_counts);
    }

    // Banner shards summary
    if (d.shard_counts) {
      document.getElementById('banner-shards').innerHTML =
        d.shard_counts.map((c,i) =>
          `<div class="banner-shard-pill">Shard ${i}<br><strong>${c}</strong></div>`
        ).join('');
    }
  } catch(e) {
    console.error('model_info error:', e);
  }
}

// ── stats ────────────────────────────────────────────────────────
async function loadStats() {
  try {
    const d = await fetch('/api/stats').then(r => r.json());
    document.getElementById('stat-total-val').textContent   = d.total_patients;
    document.getElementById('stat-cancer-val').textContent  = d.cancerous;
    document.getElementById('stat-healthy-val').textContent = d.healthy;
    document.getElementById('stat-deleted-val').textContent = d.deleted_patients;
    document.getElementById('stat-seqlen-val').textContent  = (d.primary_gene_length_bp/1000).toFixed(1)+'k';
    document.getElementById('stat-windows-val').textContent = d.total_windows.toLocaleString();
    if (d.shard_counts) updateShardBars(d.shard_counts);
  } catch(e) { console.error(e); }
}

function updateShardBars(counts) {
  maxShardSize = Math.max(...counts, 1);
  counts.forEach((c, i) => {
    const el = document.getElementById(`shard-${i}-count`);
    const fill = document.getElementById(`shard-${i}-fill`);
    if (el)   el.textContent  = `${c} patients`;
    if (fill) fill.style.width = Math.round((c / maxShardSize) * 100) + '%';
  });
}

// ── patients ─────────────────────────────────────────────────────
async function loadPatients() {
  const tbody = document.getElementById('patient-tbody');
  tbody.innerHTML = `<tr><td colspan="8" class="loading-row"><i class="fa-solid fa-circle-notch fa-spin"></i> Loading…</td></tr>`;
  try {
    allPatients = await fetch('/api/patients').then(r => r.json());
    renderPatients(allPatients);
  } catch(e) {
    tbody.innerHTML = `<tr><td colspan="8" class="loading-row" style="color:var(--red)">Failed to load patients.</td></tr>`;
  }
}

function renderPatients(list) {
  const tbody = document.getElementById('patient-tbody');
  if (!list.length) {
    tbody.innerHTML = `<tr><td colspan="8" class="loading-row">No patients match filter.</td></tr>`;
    return;
  }
  tbody.innerHTML = list.map(p => {
    const del   = p.deleted;
    const label = del
      ? `<span class="label-pill pill-deleted"><i class="fa-solid fa-user-slash"></i> Deleted</span>`
      : p.label === 'Cancerous'
        ? `<span class="label-pill pill-cancer"><i class="fa-solid fa-circle-xmark"></i> Cancerous</span>`
        : `<span class="label-pill pill-healthy"><i class="fa-solid fa-circle-check"></i> Healthy</span>`;
    const shard  = p.shard >= 0
      ? `<span class="shard-pill">S${p.shard}</span>`
      : `<span style="color:var(--text-muted);font-size:11px">test</span>`;
    const priv   = del
      ? `<span style="color:var(--green);font-size:11px;font-family:var(--font-mono)"><i class="fa-solid fa-lock"></i> Unlearned</span>`
      : `<span style="color:var(--amber);font-size:11px;font-family:var(--font-mono)"><i class="fa-solid fa-lock-open"></i> In Model</span>`;
    const acts   = del
      ? `<span style="color:var(--text-muted);font-size:12px">—</span>`
      : `<button class="btn-action btn-predict" onclick="quickPredict('${p.patient_id}')">
           <i class="fa-solid fa-magnifying-glass"></i> Predict
         </button>
         <button class="btn-action btn-delete-sm" onclick="quickForget('${p.patient_id}')" style="margin-left:4px">
           <i class="fa-solid fa-eraser"></i> Forget
         </button>`;
    return `<tr class="${del?'deleted-row':''}">
      <td><span class="pid-mono">${p.patient_id}</span></td>
      <td>${label}</td>
      <td>
        <div class="gc-bar-wrap">
          <div class="gc-bar"><div class="gc-fill" style="width:${Math.round(p.gc_content)}%"></div></div>
          <span style="font-family:var(--font-mono);font-size:11px">${p.gc_content}%</span>
        </div>
      </td>
      <td><span style="font-family:var(--font-mono);font-size:12px">${p.mutation_rate}%</span></td>
      <td><span class="shard-pill">${p.gene || 'EGFR'}</span></td>
      <td>${shard}</td>
      <td>${priv}</td>
      <td>${acts}</td>
    </tr>`;
  }).join('');
}

function filterPatients() {
  const q = document.getElementById('patient-search').value.toLowerCase();
  const f = document.getElementById('patient-filter').value;
  renderPatients(allPatients.filter(p => {
    const ms = p.patient_id.toLowerCase().includes(q);
    const mf = f==='all' ? true : f==='deleted' ? p.deleted : p.label===f && !p.deleted;
    return ms && mf;
  }));
}

// ── select population ────────────────────────────────────────────
async function populateSelects() {
  if (!allPatients.length) {
    allPatients = await fetch('/api/patients').then(r => r.json());
  }
  const active = allPatients.filter(p => !p.deleted);
  const aOpts  = active.map(p => `<option value="${p.patient_id}">${p.patient_id} — ${p.label} (S${p.shard>=0?p.shard:'?'})</option>`).join('');
  const allOpts= allPatients.map(p => `<option value="${p.patient_id}">${p.patient_id} — ${p.label}${p.deleted?' (deleted)':''}</option>`).join('');
  const fs = document.getElementById('forget-patient-select');
  if (fs) fs.innerHTML = '<option value="">Select patient…</option>' + aOpts;
  const qs = document.getElementById('quick-patient');
  if (qs) qs.innerHTML = '<option value="">— Choose a patient —</option>' + aOpts;
  const ms = document.getElementById('mia-patient-select');
  if (ms) ms.innerHTML = '<option value="">Select patient for MIA check…</option>' + allOpts;
}

// ── analyze: method switch ────────────────────────────────────────
function switchMethod(method, btn) {
  document.querySelectorAll('.method-tab').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.method-panel').forEach(p => p.classList.remove('active'));
  document.getElementById(`method-${method}`).classList.add('active');
  resetResults();
}

function resetResults() {
  document.getElementById('analyze-result').innerHTML = `
    <div class="results-placeholder">
      <i class="fa-solid fa-flask"></i>
      <p>Analyze a sequence or select a patient<br>to see cancer risk prediction results here.</p>
    </div>`;
}

// ── input mode switch ─────────────────────────────────────────────
let currentInputMode = 'text';
let selectedFastaFile = null;

function setInputMode(mode) {
  currentInputMode = mode;
  const textPanel = document.getElementById('input-text-panel');
  const filePanel = document.getElementById('input-file-panel');
  const textBtn   = document.getElementById('mode-text-btn');
  const fileBtn   = document.getElementById('mode-file-btn');
  if (mode === 'text') {
    textPanel.style.display = '';
    filePanel.style.display = 'none';
    textBtn.style.borderColor  = 'var(--primary)';
    textBtn.style.color        = 'var(--primary)';
    textBtn.style.fontWeight   = '600';
    fileBtn.style.borderColor  = '';
    fileBtn.style.color        = '';
    fileBtn.style.fontWeight   = '';
  } else {
    textPanel.style.display = 'none';
    filePanel.style.display = '';
    fileBtn.style.borderColor  = 'var(--primary)';
    fileBtn.style.color        = 'var(--primary)';
    fileBtn.style.fontWeight   = '600';
    textBtn.style.borderColor  = '';
    textBtn.style.color        = '';
    textBtn.style.fontWeight   = '';
  }
}

function handleFastaFile(input) {
  selectedFastaFile = input.files[0] || null;
  const label = document.getElementById('fasta-file-name');
  if (selectedFastaFile) {
    label.textContent = selectedFastaFile.name;
    label.style.color = 'var(--primary)';
    document.getElementById('fasta-drop-zone').style.borderColor = 'var(--primary)';
  } else {
    label.textContent = 'Click or drag .fasta / .fa file here';
    label.style.color = '';
    document.getElementById('fasta-drop-zone').style.borderColor = '';
  }
}

function clearAnalyzeInput() {
  document.getElementById('seq-input').value = '';
  selectedFastaFile = null;
  const label = document.getElementById('fasta-file-name');
  if (label) label.textContent = 'Click or drag .fasta / .fa file here';
  const dz = document.getElementById('fasta-drop-zone');
  if (dz) dz.style.borderColor = '';
  const fi = document.getElementById('fasta-file-input');
  if (fi) fi.value = '';
  resetResults();
}

// ── analyze: run ──────────────────────────────────────────────────
async function analyzeSequence() {
  setResultsLoading();
  try {
    let resp;
    if (currentInputMode === 'file') {
      if (!selectedFastaFile) { setResultsError('Please select a FASTA file.'); return; }
      const formData = new FormData();
      formData.append('fasta_file', selectedFastaFile);
      resp = await fetch('/api/classify', { method: 'POST', body: formData });
    } else {
      const seq = document.getElementById('seq-input').value.trim();
      if (!seq) { setResultsError('Please enter a DNA sequence.'); return; }
      if (seq.length < 4) { setResultsError('Sequence must be at least 4 bp.'); return; }
      resp = await fetch('/api/classify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sequence: seq })
      });
    }
    const d = await resp.json();
    if (d.error) { setResultsError(d.error); return; }
    renderResult(d);
  } catch(e) { setResultsError('Network error: ' + e.message); }
}

async function predictPatient() {
  const pid = document.getElementById('quick-patient').value;
  if (!pid) { showError('Please select a patient.'); return; }
  setResultsLoading();
  try {
    const d = await fetch(`/api/predict/${pid}`).then(r => r.json());
    if (d.error) { setResultsError(d.error); return; }
    renderResult(d);
  } catch(e) { setResultsError('Network error: '+e.message); }
}

function setResultsLoading() {
  document.getElementById('analyze-result').innerHTML = `
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:3rem;gap:14px;color:var(--text-tertiary)">
      <i class="fa-solid fa-circle-notch fa-spin" style="font-size:28px;color:var(--primary)"></i>
      <span style="font-size:13px">Running CNN inference…</span>
    </div>`;
}

function setResultsError(msg) {
  document.getElementById('analyze-result').innerHTML = `
    <div style="padding:1.5rem">
      <div style="background:var(--red-light);border:1px solid #fecaca;border-radius:var(--radius-sm);padding:1rem;color:var(--red);font-size:13px;display:flex;gap:8px">
        <i class="fa-solid fa-circle-xmark" style="margin-top:1px"></i><span>${msg}</span>
      </div>
    </div>`;
}

function renderResult(d) {
  const cancer = d.prediction === 'Cancerous';
  const prob   = Math.round((d.cancer_probability||0)*100);
  const conf   = Math.round((d.confidence||0)*100);
  const model  = d.model_used || 'unknown';
  const modelTag = model === 'SISA_CNN'
    ? `<span class="result-model-tag tag-cnn"><i class="fa-solid fa-brain"></i> SISA CNN</span>`
    : `<span class="result-model-tag tag-heuristic"><i class="fa-solid fa-triangle-exclamation"></i> Heuristic</span>`;

  const patRow = d.patient_id ? `
    <div class="metric-box" style="grid-column:1/-1;display:flex;justify-content:space-between;align-items:center;background:var(--blue-50);border-color:var(--primary-border)">
      <span style="font-size:11px;text-transform:uppercase;letter-spacing:.7px;color:var(--text-muted);font-family:var(--font-mono)">Patient ID</span>
      <span class="pid-mono" style="font-size:14px">${d.patient_id}</span>
    </div>` : '';

  const trueLabelRow = d.true_label ? `
    <div class="metric-box" style="background:${d.true_label==='Cancerous'?'var(--red-light)':'var(--green-light)'}">
      <div class="metric-label">True Label</div>
      <div class="metric-val" style="font-size:14px;color:${d.true_label==='Cancerous'?'var(--red)':'var(--green)'}">${d.true_label}</div>
    </div>` : '';

  const extraRows = d.length
    ? `<div class="metric-box"><div class="metric-label">Seq Length</div><div class="metric-val">${d.length.toLocaleString()} <span style="font-size:12px;color:var(--text-muted)">bp</span></div></div>
       <div class="metric-box"><div class="metric-label">GC Content</div><div class="metric-val">${d.gc_content}%</div></div>`
    : `<div class="metric-box"><div class="metric-label">GC Content</div><div class="metric-val">${d.gc_content||'—'}%</div></div>
       <div class="metric-box"><div class="metric-label">Mutation Rate</div><div class="metric-val">${d.mutation_rate||'—'}%</div></div>`;

  document.getElementById('analyze-result').innerHTML = `
    <div class="pred-verdict ${cancer?'cancerous':'healthy'}">
      <div class="pred-icon">${cancer?'🧬':'✅'}</div>
      <div style="flex:1">
        <div class="pred-label" style="color:${cancer?'var(--red)':'var(--green)'}">${d.prediction}</div>
        <div class="pred-conf">${conf}% model confidence</div>
      </div>
      ${modelTag}
    </div>

    <div class="prob-bar-wrap">
      <div class="prob-bar-label">
        <span>Cancer Probability</span>
        <strong style="color:${cancer?'var(--red)':'var(--green)'}">${prob}%</strong>
      </div>
      <div class="prob-bar">
        <div class="prob-fill ${prob>50?'high':'low'}" style="width:${prob}%"></div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:10px;color:var(--text-muted);margin-top:3px;font-family:var(--font-mono)">
        <span>0% Healthy</span><span>50%</span><span>100% Cancerous</span>
      </div>
    </div>

    <div class="pred-metrics">
      ${patRow}
      <div class="metric-box"><div class="metric-label">Confidence</div><div class="metric-val">${conf}%</div></div>
      ${trueLabelRow || `<div class="metric-box"><div class="metric-label">Cancer Prob.</div><div class="metric-val">${prob}%</div></div>`}
      ${extraRows}
      <div class="metric-box"><div class="metric-label">Gene</div><div class="metric-val">${d.gene || 'EGFR'}</div></div>
    </div>

    <div style="background:var(--blue-50);border:1px solid var(--primary-border);border-radius:var(--radius-sm);padding:10px 14px;font-size:12px;color:var(--text-secondary);display:flex;gap:8px">
      <i class="fa-solid fa-brain" style="color:var(--primary);margin-top:1px"></i>
      <span>${cancer
        ? 'High k-mer diversity detected — mutation patterns consistent with oncogenic EGFR variants.'
        : 'Low k-mer diversity — sequence entropy consistent with normal EGFR genomic profile.'}</span>
    </div>`;
}


// ── unlearn ───────────────────────────────────────────────────────
async function forgetPatient() {
  const pid    = document.getElementById('forget-patient-select').value;
  const reason = document.getElementById('forget-reason').value;
  if (!pid) { showError('Please select a patient.'); return; }

  const btn = document.getElementById('forget-btn');
  btn.disabled = true;
  btn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Retraining shard…';

  const status = document.getElementById('unlearn-status');
  status.className = 'unlearn-status';
  status.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> Executing SISA unlearning for <strong>${pid}</strong>…`;
  status.classList.remove('hidden');

  const shardId = hashShard(pid, 4);
  highlightShard(shardId);

  try {
    const d = await fetch('/api/forget', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({patient_id: pid, reason})
    }).then(r => r.json());

    btn.disabled = false;
    btn.innerHTML = '<i class="fa-solid fa-eraser"></i> Execute Unlearning';

    if (d.success) {
      const det  = d.details;
      const real = det.real_unlearning;
      status.className = 'unlearn-status success';
      status.innerHTML = `
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;font-size:14px;font-weight:600">
          <i class="fa-solid fa-circle-check"></i>
          Patient <strong>${pid}</strong> successfully unlearned
          ${real ? '<span style="margin-left:6px;font-size:11px;background:rgba(5,150,105,.15);color:#059669;padding:2px 8px;border-radius:20px;font-weight:500">REAL SISA</span>'
                 : '<span style="margin-left:6px;font-size:11px;background:var(--amber-light);color:var(--amber);padding:2px 8px;border-radius:20px;font-weight:500">Simulated</span>'}
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;font-size:12px;font-family:var(--font-mono)">
          <div style="background:rgba(5,150,105,.08);padding:7px 10px;border-radius:6px">
            <div style="opacity:.7;margin-bottom:2px">Shard retrained</div>
            <strong>Shard ${det.shard_retrained}</strong>
          </div>
          <div style="background:rgba(5,150,105,.08);padding:7px 10px;border-radius:6px">
            <div style="opacity:.7;margin-bottom:2px">Unlearn time</div>
            <strong>${det.unlearn_time_s}s</strong>
          </div>
          <div style="background:rgba(5,150,105,.08);padding:7px 10px;border-radius:6px">
            <div style="opacity:.7;margin-bottom:2px">Full retrain est.</div>
            <strong>${det.full_retrain_est_s}s</strong>
          </div>
          <div style="background:rgba(5,150,105,.1);padding:7px 10px;border-radius:6px">
            <div style="opacity:.7;margin-bottom:2px">Speedup</div>
            <strong style="color:#059669;font-size:16px">${det.speedup}×</strong>
          </div>
        </div>
        <div style="margin-top:10px;font-size:12px;display:flex;gap:8px;align-items:center;background:var(--blue-50);padding:8px 12px;border-radius:6px;border:1px solid var(--primary-border)">
          <i class="fa-solid fa-user-secret" style="color:var(--primary)"></i>
          MIA confidence after unlearning: <strong style="font-family:var(--font-mono)">${det.mia_confidence_after}</strong>
          &nbsp;&mdash;&nbsp;<strong style="color:${det.mia_confidence_after<.65?'var(--green)':'var(--amber)'}">${det.verdict}</strong>
          ${real ? '<span style="opacity:.7;font-size:11px">(real softmax)</span>' : '<span style="opacity:.7;font-size:11px">(simulated)</span>'}
        </div>`;

      // Update shard bars with live counts from server
      if (d.shard_counts) updateShardBars(d.shard_counts);
      if (det.speedup) {
        document.getElementById('speedup-val').textContent = `${det.speedup}×`;
        document.getElementById('speedup-sub').textContent =
          `Shard ${det.shard_retrained} retrained in ${det.unlearn_time_s}s — ${det.full_retrain_est_s}s saved`;
      }

      const p = allPatients.find(x => x.patient_id === pid);
      if (p) p.deleted = true;
      await populateSelects();
      await loadStats();
    } else {
      status.className = 'unlearn-status error';
      status.innerHTML = `<i class="fa-solid fa-circle-xmark"></i> ${d.error || 'Unlearning failed.'}`;
    }
  } catch(e) {
    btn.disabled = false;
    btn.innerHTML = '<i class="fa-solid fa-eraser"></i> Execute Unlearning';
    status.className = 'unlearn-status error';
    status.innerHTML = `<i class="fa-solid fa-circle-xmark"></i> Error: ${e.message}`;
  }
  setTimeout(() => unhighlightShards(), 5000);
}

function hashShard(pid, n) {
  let h = 0;
  for (const c of pid) h = (h*31 + c.charCodeAt(0)) & 0xffffffff;
  return Math.abs(h) % n;
}
function highlightShard(id) {
  document.querySelectorAll('.shard-card').forEach(c => c.classList.remove('active-shard'));
  const s = document.getElementById(`shard-${id}`);
  if (s) { s.classList.add('active-shard'); s.querySelector('.shard-title').textContent = `Shard ${id} 🔄`; }
}
function unhighlightShards() {
  document.querySelectorAll('.shard-card').forEach((c,i) => {
    c.classList.remove('active-shard');
    c.querySelector('.shard-title').textContent = `Shard ${i}`;
  });
}

// ── quick actions from patient table ─────────────────────────────
async function quickPredict(pid) {
  showView('analyze');
  await populateSelects();
  document.querySelectorAll('.method-tab')[1].click();
  document.getElementById('quick-patient').value = pid;
  predictPatient();
}
async function quickForget(pid) {
  showView('unlearn');
  await populateSelects();
  document.getElementById('forget-patient-select').value = pid;
}

// ── audit log ─────────────────────────────────────────────────────
async function loadAuditLog() {
  const tbody = document.getElementById('audit-tbody');
  try {
    const log = await fetch('/api/unlearn_log').then(r => r.json());
    if (!log.length) {
      tbody.innerHTML = `<tr><td colspan="10" class="loading-row">No unlearning events yet.</td></tr>`;
      return;
    }
    tbody.innerHTML = log.slice().reverse().map(e => {
      const real = e.real_unlearning;
      return `<tr>
        <td><span class="pid-mono">${e.event_id}</span></td>
        <td><span class="pid-mono">${e.patient_id}</span></td>
        <td style="text-align:center;font-family:var(--font-mono)">Shard ${e.shard_retrained}</td>
        <td style="font-family:var(--font-mono)">${e.unlearn_time_s}s</td>
        <td style="font-family:var(--font-mono)">${e.full_retrain_est_s}s</td>
        <td style="color:var(--green);font-family:var(--font-mono);font-weight:600">${e.speedup}×</td>
        <td style="font-family:var(--font-mono)">${e.mia_confidence_after}</td>
        <td><span class="label-pill pill-healthy"><i class="fa-solid fa-check"></i> ${e.verdict}</span></td>
        <td>${real
          ? `<span class="label-pill" style="background:rgba(5,150,105,.1);color:#059669">Real</span>`
          : `<span class="label-pill" style="background:var(--amber-light);color:var(--amber)">Sim</span>`}</td>
        <td style="color:var(--text-muted);font-size:11px;font-family:var(--font-mono)">${e.timestamp}</td>
      </tr>`;
    }).join('');
  } catch(e) {
    tbody.innerHTML = `<tr><td colspan="10" class="loading-row" style="color:var(--red)">Failed to load log.</td></tr>`;
  }
}

// ── MIA ───────────────────────────────────────────────────────────
async function runMIA() {
  const pid = document.getElementById('mia-patient-select').value;
  if (!pid) { showError('Please select a patient.'); return; }
  const div = document.getElementById('mia-result');
  div.className = 'mia-result';
  div.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> Running MIA on ${pid}…`;
  div.classList.remove('hidden');
  try {
    const d = await fetch(`/api/mia/${pid}`).then(r => r.json());
    const forgotten = d.mia_confidence < 0.60;
    div.className = `mia-result ${forgotten?'mia-forgotten':'mia-remembered'}`;
    div.innerHTML = `
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
        <i class="fa-solid ${forgotten?'fa-user-slash':'fa-user'}" style="font-size:20px"></i>
        <strong style="font-size:14px">${d.verdict}</strong>
        ${d.real_mia
          ? `<span style="font-size:11px;opacity:.7">(real softmax confidence)</span>`
          : `<span style="font-size:11px;opacity:.7">(simulated)</span>`}
      </div>
      <div class="mia-conf">${d.mia_confidence}</div>
      <div style="font-size:12px;margin-top:6px;line-height:1.5;opacity:.85">${d.interpretation}</div>`;
  } catch(e) {
    div.className = 'mia-result mia-remembered';
    div.innerHTML = `Error: ${e.message}`;
  }
}

// ── modal ─────────────────────────────────────────────────────────
function showModal(html) {
  document.getElementById('modal-content').innerHTML = html;
  document.getElementById('modal-overlay').classList.remove('hidden');
}
function closeModal() {
  document.getElementById('modal-overlay').classList.add('hidden');
}
function showError(msg) {
  showModal(`
    <div style="text-align:center;padding:1rem">
      <div style="width:50px;height:50px;background:var(--amber-light);border-radius:50%;display:flex;align-items:center;justify-content:center;margin:0 auto 1rem">
        <i class="fa-solid fa-triangle-exclamation" style="font-size:22px;color:var(--amber)"></i>
      </div>
      <p style="color:var(--text-primary);font-size:14px;margin-bottom:1.25rem">${msg}</p>
      <button class="btn-primary" onclick="closeModal()">OK</button>
    </div>`);
}

// ── init ──────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  await loadModelInfo();
  await loadStats();
  try {
    allPatients = await fetch('/api/patients').then(r => r.json());
  } catch(e) { console.error(e); }
  await populateSelects();
});
