const API_URL = "http://localhost:8000/predict";

async function sendBlob(blob, onResult) {
  const form = new FormData();
  form.append('file', blob, 'capture.jpg');
  const res = await fetch(API_URL, { method: 'POST', body: form });
  const json = await res.json();
  onResult(json);
}

// Webcam
const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');

navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
  video.srcObject = stream;
});

document.getElementById('captureBtn').addEventListener('click', async () => {
  const c = document.createElement('canvas');
  c.width = video.videoWidth || 480;
  c.height = video.videoHeight || 360;
  const cc = c.getContext('2d');
  cc.drawImage(video, 0, 0, c.width, c.height);
  c.toBlob(async (blob) => {
    await sendBlob(blob, (json) => {
      drawResults(c, json.predictions);
    });
  }, 'image/jpeg');
});

function drawResults(canvasElem, predictions) {
  const displayCanvas = overlay;
  const dctx = displayCanvas.getContext('2d');
  dctx.clearRect(0, 0, displayCanvas.width, displayCanvas.height);
  const img = new Image();
  img.onload = () => {
    dctx.drawImage(img, 0, 0, displayCanvas.width, displayCanvas.height);
    predictions.forEach(p => {
      const [x1, y1, x2, y2] = p.box;
      // scale coordinates from original image size to canvas size (assumes same size)
      dctx.strokeStyle = p.label === 'mask' ? 'green' : 'red';
      dctx.lineWidth = 3;
      dctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      dctx.fillStyle = dctx.strokeStyle;
      dctx.fillText(`${p.label} (${(p.confidence*100).toFixed(0)}%)`, x1 + 4, y1 + 14);
    });
  };
  canvasElem.toDataURL && (img.src = canvasElem.toDataURL('image/jpeg'));
}

// Upload
const fileInput = document.getElementById('fileInput');
const uploadCanvas = document.getElementById('uploadCanvas');
const uctx = uploadCanvas.getContext('2d');
fileInput.addEventListener('change', (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const img = new Image();
  const url = URL.createObjectURL(f);
  img.onload = () => {
    uploadCanvas.width = img.width > 480 ? 480 : img.width;
    const h = img.height * (uploadCanvas.width / img.width);
    uploadCanvas.height = h;
    uctx.drawImage(img, 0, 0, uploadCanvas.width, uploadCanvas.height);
    uploadCanvas.toBlob(async (blob) => {
      await sendBlob(blob, (json) => {
        // draw on upload canvas
        json.predictions.forEach(p => {
          uctx.strokeStyle = p.label === 'mask' ? 'green' : 'red';
          uctx.lineWidth = 3;
          const [x1,y1,x2,y2] = p.box;
          uctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
          uctx.fillStyle = uctx.strokeStyle;
          uctx.fillText(`${p.label} (${(p.confidence*100).toFixed(0)}%)`, x1 + 4, y1 + 14);
        });
      });
    }, 'image/jpeg');
  };
  img.src = url;
});
