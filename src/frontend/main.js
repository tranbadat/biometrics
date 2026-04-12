const API = 'http://localhost:8000';
const MIN_GOOD_PHOTOS  = 15;   // số ảnh tốt tối thiểu để enroll
const SHARP_THRESHOLD  = 280;  // Laplacian variance threshold
const WARN_THRESHOLD   = 120;

// ── Tab switching ────────────────────────────────────────────────────────────
function switchTab(name, btn) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  btn.classList.add('active');

  if (name === 'recog') startRecogCamera();
  if (name === 'enroll') startEnrollCamera();
}

// ── Toast ────────────────────────────────────────────────────────────────────
let _toastTimer;
function showToast(msg, type = '') {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'show ' + type;
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => { t.className = ''; }, 3500);
}

// ── Sharpness check (Laplacian variance, client-side) ────────────────────────
function getSharpness(canvas) {
  const tmp = document.createElement('canvas');
  tmp.width = 120; tmp.height = 90;
  tmp.getContext('2d').drawImage(canvas, 0, 0, 120, 90);
  const d = tmp.getContext('2d').getImageData(0, 0, 120, 90).data;
  const W = 120, H = 90;
  let sum = 0, count = 0;
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const gi = (r, c) => {
        const i = (r * W + c) * 4;
        return 0.299 * d[i] + 0.587 * d[i+1] + 0.114 * d[i+2];
      };
      const lap = gi(y-1,x) + gi(y+1,x) + gi(y,x-1) + gi(y,x+1) - 4*gi(y,x);
      sum += lap * lap; count++;
    }
  }
  return count ? sum / count : 0;
}

function qualityClass(sharpness) {
  if (sharpness >= SHARP_THRESHOLD) return 'good';
  if (sharpness >= WARN_THRESHOLD)  return 'warn';
  return 'bad';
}
function qualityLabel(sharpness) {
  if (sharpness >= SHARP_THRESHOLD) return '✓';
  if (sharpness >= WARN_THRESHOLD)  return '~';
  return '✗';
}

// ════════════════════════════════════════════════════════════════════════════
// TAB 1 — ENROLL
// ════════════════════════════════════════════════════════════════════════════
let enrollStream = null;
let enrollPhotos = [];   // [{blob, dataURL, sharpness}]

async function startEnrollCamera() {
  if (enrollStream) return;
  try {
    enrollStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
    });
    const v = document.getElementById('enroll-video');
    v.srcObject = enrollStream;
  } catch (e) {
    showToast('Không truy cập được camera: ' + e.message, 'error');
  }
}

function enrollCapture() {
  const video = document.getElementById('enroll-video');
  const W = video.videoWidth  || 400;
  const H = video.videoHeight || 300;

  const c = document.createElement('canvas');
  c.width = W; c.height = H;
  c.getContext('2d').drawImage(video, 0, 0, W, H);

  const sharpness = getSharpness(c);
  const cls       = qualityClass(sharpness);

  c.toBlob(blob => {
    const dataURL = c.toDataURL('image/jpeg', 0.9);
    enrollPhotos.push({ blob, dataURL, sharpness, cls });
    renderEnrollThumbs();
  }, 'image/jpeg', 0.9);
}

function renderEnrollThumbs() {
  const grid  = document.getElementById('enroll-thumb-grid');
  const empty = document.getElementById('enroll-empty');
  const fill  = document.getElementById('enroll-progress-fill');
  const pLabel = document.getElementById('enroll-progress-label');
  const qLabel = document.getElementById('enroll-quality-label');
  const saveBtn = document.getElementById('enroll-save-btn');

  // Xóa thumbs cũ (giữ empty state)
  [...grid.querySelectorAll('.thumb-item')].forEach(el => el.remove());

  if (enrollPhotos.length === 0) {
    empty.style.display = '';
    fill.style.width = '0%';
    pLabel.textContent = `0 / ${MIN_GOOD_PHOTOS} ảnh tốt`;
    qLabel.textContent = '—';
    saveBtn.disabled = true;
    return;
  }
  empty.style.display = 'none';

  const goodCount = enrollPhotos.filter(p => p.cls === 'good').length;
  const warnCount = enrollPhotos.filter(p => p.cls === 'warn').length;
  const badCount  = enrollPhotos.filter(p => p.cls === 'bad').length;

  enrollPhotos.forEach((photo, i) => {
    const item = document.createElement('div');
    item.className = `thumb-item ${photo.cls}`;

    const img = document.createElement('img');
    img.src = photo.dataURL;

    const badge = document.createElement('div');
    badge.className = `quality-badge ${photo.cls}`;
    badge.textContent = qualityLabel(photo.sharpness);

    const rmBtn = document.createElement('button');
    rmBtn.className = 'remove-btn';
    rmBtn.textContent = '✕';
    rmBtn.onclick = () => { enrollPhotos.splice(i, 1); renderEnrollThumbs(); };

    item.append(img, badge, rmBtn);
    grid.appendChild(item);
  });

  const pct = Math.min(100, (goodCount / MIN_GOOD_PHOTOS) * 100);
  fill.style.width = pct + '%';
  fill.className = 'progress-fill' + (goodCount >= MIN_GOOD_PHOTOS ? ' done' : '');
  pLabel.textContent = `${goodCount} / ${MIN_GOOD_PHOTOS} ảnh tốt  (${enrollPhotos.length} tổng)`;
  qLabel.textContent = `✓${goodCount}  ~${warnCount}  ✗${badCount}`;

  const canSave = goodCount >= MIN_GOOD_PHOTOS
    && document.getElementById('enroll-name').value.trim()
    && document.getElementById('enroll-id').value.trim();
  saveBtn.disabled = !canSave;
}

// event listeners gắn trong DOMContentLoaded (xem cuối file)

function enrollClearAll() {
  enrollPhotos = [];
  renderEnrollThumbs();
}

async function enrollSave() {
  const name = document.getElementById('enroll-name').value.trim();
  const uid  = document.getElementById('enroll-id').value.trim();

  if (!name || !uid) { showToast('Vui lòng nhập họ tên và mã người dùng', 'error'); return; }

  const goodPhotos = enrollPhotos.filter(p => p.cls !== 'bad');
  if (goodPhotos.length < MIN_GOOD_PHOTOS) {
    showToast(`Cần ít nhất ${MIN_GOOD_PHOTOS} ảnh có chất lượng tốt/trung bình`, 'error');
    return;
  }

  const btn = document.getElementById('enroll-save-btn');
  const status = document.getElementById('enroll-status');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Đang xử lý...';
  status.textContent = '';

  const form = new FormData();
  form.append('name', name);
  form.append('user_id', uid);
  goodPhotos.forEach((p, i) => form.append('files', p.blob, `frame_${String(i).padStart(4,'0')}.jpg`));

  try {
    const res  = await fetch(`${API}/enroll`, { method: 'POST', body: form });
    const json = await res.json();

    if (!res.ok) throw new Error(json.detail || 'Lỗi server');

    showToast(`✅ Đăng ký thành công "${name}" (${json.saved} ảnh)`, 'success');
    status.textContent = `✓ Đã lưu ${json.saved} ảnh vào data/faces/${uid}/`;
    enrollPhotos = [];
    renderEnrollThumbs();
    document.getElementById('enroll-name').value = '';
    document.getElementById('enroll-id').value = '';
  } catch (e) {
    showToast('Lỗi: ' + e.message, 'error');
    status.textContent = '✗ ' + e.message;
  } finally {
    btn.disabled = false;
    btn.innerHTML = '💾 Lưu và Đăng ký';
    renderEnrollThumbs();
  }
}

// ════════════════════════════════════════════════════════════════════════════
// TAB 2 — NHẬN DIỆN
// ════════════════════════════════════════════════════════════════════════════
let recogStream   = null;
let recogMode     = 'camera';   // 'camera' | 'upload'

async function startRecogCamera() {
  if (recogStream) return;
  try {
    recogStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
    });
    document.getElementById('recog-video').srcObject = recogStream;
  } catch (e) {
    showToast('Không truy cập được camera: ' + e.message, 'error');
  }
}

function setInputMode(mode) {
  recogMode = mode;
  document.getElementById('recog-camera-mode').style.display = mode === 'camera' ? '' : 'none';
  document.getElementById('recog-upload-mode').style.display = mode === 'upload' ? '' : 'none';
  document.getElementById('toggle-cam').classList.toggle('active', mode === 'camera');
  document.getElementById('toggle-upload').classList.toggle('active', mode === 'upload');
}

async function recogCapture() {
  const video = document.getElementById('recog-video');
  const W = video.videoWidth  || 560;
  const H = video.videoHeight || 420;

  const capture = document.createElement('canvas');
  capture.width = W; capture.height = H;
  capture.getContext('2d').drawImage(video, 0, 0, W, H);

  capture.toBlob(async blob => {
    await runPredict(blob, W, H);
  }, 'image/jpeg', 0.92);
}

async function recogUpload(input) {
  const file = input.files[0];
  if (!file) return;

  // Decode ảnh, resize về tối đa 640px để tiết kiệm bandwidth
  const img = new window.Image();
  const url = URL.createObjectURL(file);
  img.onload = async () => {
    const MAX = 640;
    const scale = img.width > MAX ? MAX / img.width : 1;
    const W = Math.round(img.width  * scale);
    const H = Math.round(img.height * scale);

    const c = document.createElement('canvas');
    c.width = W; c.height = H;
    c.getContext('2d').drawImage(img, 0, 0, W, H);
    URL.revokeObjectURL(url);

    c.toBlob(async blob => {
      await runPredict(blob, W, H);
    }, 'image/jpeg', 0.92);
  };
  img.src = url;
  input.value = '';
}

// imgW, imgH: kích thước ảnh gửi lên backend — box coords trong không gian này
async function runPredict(blob, imgW, imgH) {
  const form = new FormData();
  form.append('file', blob, 'frame.jpg');

  // Hiển thị ảnh capture lên canvas ngay (trước khi có response)
  const resultCanvas = document.getElementById('recog-result-canvas');
  const ctx = resultCanvas.getContext('2d');
  resultCanvas.width  = imgW;
  resultCanvas.height = imgH;
  resultCanvas.style.display = 'block';

  const bmp = await createImageBitmap(blob);
  ctx.drawImage(bmp, 0, 0);

  document.getElementById('recog-waiting').style.display = 'none';
  document.getElementById('recog-result-detail').style.display = 'none';

  try {
    const res  = await fetch(`${API}/predict`, { method: 'POST', body: form });
    const json = await res.json();
    if (!res.ok) throw new Error(json.detail || 'Lỗi server');

    const preds = json.predictions || [];

    // Vẽ lại ảnh sạch rồi vẽ boxes — imgW × imgH khớp canvas → không cần scale
    ctx.drawImage(bmp, 0, 0);
    drawBoxes(ctx, preds, imgW, imgH);

    // Cập nhật panel kết quả
    showRecogResult(preds, imgW, imgH);

    // Lưu vào log
    addLog(resultCanvas, preds);

  } catch (e) {
    showToast('Lỗi: ' + e.message, 'error');
    document.getElementById('recog-waiting').style.display = '';
  }
}

// Vẽ boxes — coords từ backend đã ở cùng không gian imgW × imgH = canvas size
function drawBoxes(ctx, preds, imgW, imgH) {
  const scaleX = ctx.canvas.width  / imgW;
  const scaleY = ctx.canvas.height / imgH;

  ctx.font = 'bold 13px "Segoe UI", sans-serif';
  ctx.lineWidth = 2;

  preds.forEach(p => {
    const [x1, y1, x2, y2] = p.box;
    const sx = x1 * scaleX, sy = y1 * scaleY;
    const sw = (x2 - x1) * scaleX, sh = (y2 - y1) * scaleY;

    const color = p.label === 'mask' ? '#22c55e' : '#ef4444';
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.strokeRect(sx, sy, sw, sh);

    // Corner accent
    const cs = Math.min(sw, sh) * 0.18;
    ctx.lineWidth = 4;
    [[sx, sy], [sx+sw, sy], [sx, sy+sh], [sx+sw, sy+sh]].forEach(([cx, cy], i) => {
      ctx.beginPath();
      ctx.moveTo(cx + (i%2===0 ? cs : -cs), cy);
      ctx.lineTo(cx, cy);
      ctx.lineTo(cx, cy + (i<2 ? cs : -cs));
      ctx.stroke();
    });

    // Label pill
    const maskTxt = p.label === 'mask' ? '😷 Đeo khẩu trang' : '😶 Không đeo';
    const idTxt   = p.identity ? `  |  👤 ${p.identity}` : '';
    const label   = maskTxt + idTxt;
    ctx.lineWidth = 1;
    const tw = ctx.measureText(label).width;
    const ph = 22, pw = tw + 12;
    const lx = sx, ly = sy > 26 ? sy - 28 : sy + sh + 6;

    ctx.fillStyle = 'rgba(0,0,0,0.72)';
    roundRect(ctx, lx, ly, pw, ph, 4);
    ctx.fill();

    ctx.fillStyle = color;
    ctx.fillText(label, lx + 6, ly + 15);
  });
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

function showRecogResult(preds, imgW, imgH) {
  const detail = document.getElementById('recog-result-detail');
  const waiting = document.getElementById('recog-waiting');

  if (preds.length === 0) {
    waiting.style.display = '';
    waiting.innerHTML = '<div class="icon">😶</div>Không phát hiện khuôn mặt nào trong ảnh';
    return;
  }

  detail.style.display = 'block';
  waiting.style.display = 'none';

  // Lấy prediction đầu tiên (hoặc confidence cao nhất)
  const p = preds.slice().sort((a,b) => b.confidence - a.confidence)[0];

  document.getElementById('res-identity').textContent = p.identity || 'Chưa nhận ra';
  document.getElementById('res-uid').textContent = p.identity ? `(đã enroll)` : '—';
  document.getElementById('res-face-count').textContent = `${preds.length} khuôn mặt`;

  const maskTxt = p.label === 'mask' ? '😷 Đeo khẩu trang' : '😶 Không đeo khẩu trang';
  document.getElementById('res-mask').textContent = maskTxt;

  const maskPct = (p.confidence * 100).toFixed(1) + '%';
  document.getElementById('res-mask-conf').innerHTML =
    `${maskPct} <span class="conf-bar"><span class="conf-fill" style="width:${maskPct};background:${p.label==='mask'?'#22c55e':'#ef4444'}"></span></span>`;

  const idConf = p.identity_confidence > 0
    ? (p.identity_confidence * 100).toFixed(1) + '%'
    : '—';
  document.getElementById('res-id-conf').innerHTML = p.identity_confidence > 0
    ? `${idConf} <span class="conf-bar"><span class="conf-fill" style="width:${idConf}"></span></span>`
    : '—';

  const badgesEl = document.getElementById('res-badges');
  const maskCls  = p.label === 'mask' ? 'badge-mask' : 'badge-nomask';
  const maskBadge = `<span class="${maskCls}">${p.label === 'mask' ? 'CÓ KHẨU TRANG' : 'KHÔNG KHẨU TRANG'}</span>`;
  const idBadge   = p.identity
    ? `<span class="badge-mask" style="margin-left:6px">NHẬN RA</span>`
    : `<span class="badge-unknown" style="margin-left:6px">CHƯA ENROLL</span>`;
  badgesEl.innerHTML = maskBadge + idBadge;
}

// ════════════════════════════════════════════════════════════════════════════
// TAB 3 — LOGS
// ════════════════════════════════════════════════════════════════════════════
let logs = [];

function addLog(canvas, preds) {
  const thumb = document.createElement('canvas');
  thumb.width = 90; thumb.height = 68;
  thumb.getContext('2d').drawImage(canvas, 0, 0, 90, 68);

  logs.unshift({ canvas: thumb, preds, time: new Date() });
  renderLogs();

  const count = document.getElementById('log-count');
  count.textContent = logs.length;
  count.style.display = 'inline';
}

function renderLogs() {
  const list  = document.getElementById('log-list');
  const empty = document.getElementById('log-empty');

  [...list.querySelectorAll('.log-card')].forEach(el => el.remove());

  if (logs.length === 0) {
    empty.style.display = '';
    return;
  }
  empty.style.display = 'none';

  logs.forEach((log, idx) => {
    const card = document.createElement('div');
    card.className = 'log-card';

    // Thumbnail
    const thumbWrap = document.createElement('div');
    thumbWrap.appendChild(log.canvas);
    card.appendChild(thumbWrap);

    // Info
    const info = document.createElement('div');
    info.className = 'log-info';

    const timeStr = log.time.toLocaleTimeString('vi-VN');
    const dateStr = log.time.toLocaleDateString('vi-VN');

    if (log.preds.length === 0) {
      info.innerHTML = `
        <div class="log-name" style="color:var(--muted)">Không phát hiện khuôn mặt</div>
        <div class="log-time">${dateStr} ${timeStr}</div>`;
    } else {
      const facesHtml = log.preds.map(p => {
        const maskCls  = p.label === 'mask' ? 'badge-mask' : 'badge-nomask';
        const maskText = p.label === 'mask' ? '😷 Đeo khẩu trang' : '😶 Không đeo';
        const idText   = p.identity ? `👤 ${p.identity}` : '👤 Chưa nhận ra';
        const confMask = (p.confidence * 100).toFixed(0) + '%';
        const confId   = p.identity_confidence > 0 ? (p.identity_confidence * 100).toFixed(0) + '%' : '';
        return `<div style="margin-bottom:6px">
          <span class="${maskCls}">${maskText}</span>
          <span class="log-conf" style="margin-left:8px">${confMask}</span>
          &nbsp;·&nbsp;
          <strong>${idText}</strong>
          ${confId ? `<span class="log-conf" style="margin-left:6px">${confId}</span>` : ''}
        </div>`;
      }).join('');

      info.innerHTML = `
        <div class="log-name">${log.preds.length} khuôn mặt phát hiện</div>
        <div class="log-time">${dateStr} ${timeStr}</div>
        <div class="log-meta" style="margin-top:8px;flex-direction:column;align-items:flex-start">${facesHtml}</div>`;
    }
    card.appendChild(info);

    // Action
    const action = document.createElement('div');
    const delBtn = document.createElement('button');
    delBtn.className = 'btn btn-outline';
    delBtn.style.cssText = 'padding:5px 10px;font-size:11px';
    delBtn.textContent = '🗑';
    delBtn.onclick = () => { logs.splice(idx, 1); renderLogs(); updateLogCount(); };
    action.appendChild(delBtn);
    card.appendChild(action);

    list.appendChild(card);
  });
}

function updateLogCount() {
  const count = document.getElementById('log-count');
  if (logs.length === 0) { count.style.display = 'none'; return; }
  count.textContent = logs.length;
  count.style.display = 'inline';
}

function clearLogs() {
  logs = [];
  renderLogs();
  updateLogCount();
}

// ════════════════════════════════════════════════════════════════════════════
// INIT — gắn tất cả event listener sau khi DOM sẵn sàng
// ════════════════════════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {

  // ── Tab switching ─────────────────────────────────────────────────────────
  document.getElementById('tab-btn-enroll').addEventListener('click', function () {
    switchTab('enroll', this);
  });
  document.getElementById('tab-btn-recog').addEventListener('click', function () {
    switchTab('recog', this);
  });
  document.getElementById('tab-btn-logs').addEventListener('click', function () {
    switchTab('logs', this);
  });

  // ── Tab 1: Enroll ─────────────────────────────────────────────────────────
  document.getElementById('enroll-capture-btn').addEventListener('click', enrollCapture);

  // Nút "Xóa tất cả" — tìm theo text vì không có id
  document.querySelectorAll('#tab-enroll .btn-outline').forEach(btn => {
    if (btn.textContent.includes('Xóa tất cả')) {
      btn.addEventListener('click', enrollClearAll);
    }
  });

  document.getElementById('enroll-save-btn').addEventListener('click', enrollSave);

  document.getElementById('enroll-name').addEventListener('input', renderEnrollThumbs);
  document.getElementById('enroll-id').addEventListener('input', renderEnrollThumbs);

  // ── Tab 2: Recognition ───────────────────────────────────────────────────
  document.getElementById('toggle-cam').addEventListener('click', () => setInputMode('camera'));
  document.getElementById('toggle-upload').addEventListener('click', () => setInputMode('upload'));

  // Nút "Chụp & Phân tích"
  document.querySelector('#recog-camera-mode .btn-primary').addEventListener('click', recogCapture);

  document.getElementById('recog-file-input').addEventListener('change', function () {
    recogUpload(this);
  });

  // ── Tab 3: Logs ──────────────────────────────────────────────────────────
  document.querySelector('#tab-logs .btn-outline').addEventListener('click', clearLogs);

  // ── Khởi động camera mặc định (tab enroll) ───────────────────────────────
  startEnrollCamera();
});
