const EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"];
const TARGET_FPS = 8;

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const statusEl = document.getElementById("status");
const emotionEl = document.getElementById("emotion");
const barsEl = document.getElementById("bars");
const fpsEl = document.getElementById("fps");

const ctx = overlay.getContext("2d");
const sender = document.createElement("canvas");
const senderCtx = sender.getContext("2d");

let stream = null;
let ws = null;
let sendTimer = null;
let inFlight = false;
let frameTimes = [];

for (const e of EMOTIONS) {
  const row = document.createElement("div");
  row.className = "bar-row";

  const name = document.createElement("div");
  name.className = "name";
  name.textContent = e;

  const barBg = document.createElement("div");
  barBg.className = "bar-bg";
  const barFill = document.createElement("div");
  barFill.className = "bar-fill";
  barFill.dataset.e = e;
  barFill.id = `bar-${e}`;
  barBg.appendChild(barFill);

  const pct = document.createElement("div");
  pct.className = "pct";
  pct.id = `pct-${e}`;
  pct.textContent = "0%";

  row.append(name, barBg, pct);
  barsEl.appendChild(row);
}

function setStatus(text) { statusEl.textContent = text; }

function resizeOverlay() {
  const rect = video.getBoundingClientRect();
  overlay.width = rect.width;
  overlay.height = rect.height;
}

function drawBox(box) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  if (!box) return;
  const [x, y, w, h] = box;
  const sx = overlay.width / video.videoWidth;
  const sy = overlay.height / video.videoHeight;
  ctx.strokeStyle = "#6ee7b7";
  ctx.lineWidth = 3;
  ctx.strokeRect(x * sx, y * sy, w * sx, h * sy);
}

function updateUI(payload) {
  if (payload.error) { setStatus(`error: ${payload.error}`); return; }
  emotionEl.textContent = payload.emotion;
  for (const [name, prob] of Object.entries(payload.probabilities)) {
    const bar = document.getElementById(`bar-${name}`);
    const pct = document.getElementById(`pct-${name}`);
    if (bar && pct) {
      bar.style.width = `${(prob * 100).toFixed(0)}%`;
      pct.textContent = `${(prob * 100).toFixed(0)}%`;
    }
  }
  drawBox(payload.face_box);

  const now = performance.now();
  frameTimes.push(now);
  frameTimes = frameTimes.filter(t => now - t < 1000);
  fpsEl.textContent = `${frameTimes.length} fps`;
}

async function captureAndSend() {
  if (!ws || ws.readyState !== WebSocket.OPEN || inFlight) return;
  if (!video.videoWidth) return;

  sender.width = video.videoWidth;
  sender.height = video.videoHeight;
  senderCtx.drawImage(video, 0, 0);

  inFlight = true;
  sender.toBlob(async (blob) => {
    if (!blob || !ws || ws.readyState !== WebSocket.OPEN) { inFlight = false; return; }
    const buf = await blob.arrayBuffer();
    ws.send(buf);
    inFlight = false;
  }, "image/jpeg", 0.7);
}

async function start() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    video.srcObject = stream;
    await video.play();
    resizeOverlay();
  } catch (e) {
    setStatus(`camera error: ${e.message}`);
    return;
  }

  const wsUrl = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/emotion`;
  ws = new WebSocket(wsUrl);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    setStatus("live");
    startBtn.disabled = true;
    stopBtn.disabled = false;
    sendTimer = setInterval(captureAndSend, 1000 / TARGET_FPS);
  };
  ws.onmessage = (ev) => {
    try { updateUI(JSON.parse(ev.data)); } catch (_) {}
  };
  ws.onerror = () => setStatus("ws error");
  ws.onclose = () => setStatus("disconnected");
}

function stop() {
  if (sendTimer) { clearInterval(sendTimer); sendTimer = null; }
  if (ws) { ws.close(); ws = null; }
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  video.srcObject = null;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  startBtn.disabled = false;
  stopBtn.disabled = true;
  setStatus("stopped");
}

startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", stop);
window.addEventListener("resize", resizeOverlay);
