// ----------------- Globālie elementi -----------------
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

const letterDisplay = document.getElementById('letter');
const mFps = document.getElementById('mFps');
const mLat = document.getElementById('mLat');
const mPred = document.getElementById('mPred');
const minConfInput = document.getElementById('minConf');
const minConfVal = document.getElementById('minConfVal');

// ----------------- Stāvoklis / veikals -----------------
const store = { MIN_CONFIDENCE: Number(minConfInput.value) };
minConfVal.textContent = store.MIN_CONFIDENCE.toFixed(2);
minConfInput.addEventListener('input', () => {
  store.MIN_CONFIDENCE = Number(minConfInput.value);
  minConfVal.textContent = store.MIN_CONFIDENCE.toFixed(2);
});

// ----------------- Modelis -----------------
let trainedModel = null;
let labelSet = [];    // klašu secība no model.json
let smoothScores = null;

// Stabilizācija
const labelWindow = [];
const WINDOW_SIZE = 15;
const ALPHA = 0.35;

// HUD / stāvoklis
let lastLetter = '?';
let latestLandmarks = null;

// ----------------- Palīgfunkcijas -----------------
function expSmooth(prev, next, a = ALPHA) { return a * next + (1 - a) * prev; }
function majority(arr) {
  if (!arr.length) return null;
  const m = new Map();
  for (const v of arr) m.set(v, (m.get(v) || 0) + 1);
  let best = null, cnt = -1;
  for (const [k, v] of m) if (v > cnt) { best = k; cnt = v; }
  return best;
}

const frameCounter = (() => {
  let acc = 0, n = 0, lastFps = 0, lastLat = 0;
  return {
    get fps() { return lastFps; },
    get latency() { return lastLat; },
    tick(dur) {
      n++; acc += dur;
      if (n >= 10) {
        const avg = acc / n; acc = 0; n = 0;
        const fps = 1000 / avg;
        lastFps = Math.round(expSmooth(lastFps || fps, fps, 0.25));
        lastLat = expSmooth(lastLat || dur, dur, 0.25);
      }
    }
  };
})();

function flattenLandmarks(landmarks) { return landmarks.flatMap(p => [p.x, p.y, p.z]); }

function speak(text) {
  if (!text) return;
  const now = performance.now();
  speak._last = speak._last || { t: 0, s: '' };
  if (speak._last.s === text && (now - speak._last.t) < 1500) return;
  speak._last = { t: now, s: text };
  const synth = window.speechSynthesis;
  const utter = new SpeechSynthesisUtterance(text);
  utter.lang = 'lv-LV';
  synth.cancel();
  synth.speak(utter);
}

// ----------------- Modelis: ielāde un prognoze -----------------
async function loadPretrainedModel() {
  // vispirms nolasa raw JSON, lai dabūtu labelSet
  const res = await fetch("./model/model.json");
  if (!res.ok) throw new Error("model.json fetch failed: " + res.status);
  const raw = await res.json();

  labelSet = raw.labelSet || [];
  console.log("✅ Ielādēts labelSet:", labelSet);

  // pēc tam ielādē pašu modeli (tf.js pats izmantos weightsManifest)
  trainedModel = await tf.loadLayersModel("./model/model.json");
  smoothScores = new Array(labelSet.length).fill(0);

  console.log("✅ Modelis ielādēts. Klašu skaits:", labelSet.length);
}

function predict(landmarks) {
  if (!trainedModel || !landmarks) return { label: '?', conf: 0 };
  const flat = flattenLandmarks(landmarks);
  return tf.tidy(() => {
    const input = tf.tensor2d([flat]);
    const probs = trainedModel.predict(input);
    const vals = probs.dataSync();

    if (!smoothScores || smoothScores.length !== vals.length) smoothScores = new Array(vals.length).fill(0);
    for (let i = 0; i < vals.length; i++) smoothScores[i] = expSmooth(smoothScores[i], vals[i], ALPHA);

    const sum = smoothScores.reduce((a, b) => a + b, 0) || 1;
    const norm = smoothScores.map(v => v / sum);
    let maxI = 0, maxV = norm[0];
    for (let i = 1; i < norm.length; i++) {
      if (norm[i] > maxV) { maxV = norm[i]; maxI = i; }
    }
    const lbl = labelSet[maxI] || '?';
    return { label: lbl, conf: maxV };
  });
}

// ----------------- MediaPipe iestatīšana -----------------
const hands = new Hands({ locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}` });
hands.setOptions({
  maxNumHands: 1,
  modelComplexity: 1,
  minDetectionConfidence: 0.7,
  minTrackingConfidence: 0.7
});

hands.onResults(async (results) => {
  const t0 = performance.now();
  ctx.save();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

  if (results.multiHandLandmarks?.length) {
    const landmarks = results.multiHandLandmarks[0];
    latestLandmarks = landmarks;

    drawConnectors(ctx, landmarks, HAND_CONNECTIONS, { color: '#22d3ee', lineWidth: 3 });
    drawLandmarks(ctx, landmarks, { color: '#ff6b6b', radius: 2 });

    const { label, conf } = predict(landmarks);
    labelWindow.push(conf >= store.MIN_CONFIDENCE ? label : '?');
    if (labelWindow.length > WINDOW_SIZE) labelWindow.shift();
    const stable = majority(labelWindow.filter(x => x !== '?'));

    mPred.textContent = stable ? `${stable} (${(conf * 100 | 0)}%)` : '—';

    if (stable && stable !== lastLetter) {
      lastLetter = stable;
      letterDisplay.textContent = stable;
      highlightLetter(stable);
      speak(stable);
    }
  }

  const t1 = performance.now();
  frameCounter.tick(t1 - t0);
  mFps.textContent = frameCounter.fps.toString();
  mLat.textContent = `${Math.round(frameCounter.latency)} ms`;

  ctx.fillStyle = "rgba(0,0,0,0.55)";
  ctx.fillRect(8, 8, 190, 68);
  ctx.fillStyle = "#fff";
  ctx.font = "12px system-ui";
  ctx.fillText(`FPS: ${frameCounter.fps}`, 16, 28);
  ctx.fillText(`Latency: ${Math.round(frameCounter.latency)} ms`, 16, 46);
  ctx.fillText(`Label: ${lastLetter}`, 16, 64);

  ctx.restore();
});

// Kamera
const camera = new Camera(video, {
  onFrame: async () => { await hands.send({ image: video }); },
  width: 640, height: 480
});

// ----------------- Alfabēta režģis -----------------
const LATVIAN_ALPHABET = [
  "A","Ā","B","C","Č","D","E","Ē","F","G","Ģ","H","I","Ī","J","K","Ķ","L","Ļ","M",
  "N","Ņ","O","P","R","S","Š","T","U","Ū","V","Z","Ž"
];
const alphabetGrid = document.getElementById('alphabetGrid');
let highlighted = null;

function renderAlphabetGrid() {
  if (!alphabetGrid) return;
  alphabetGrid.innerHTML = '';
  for (const ch of LATVIAN_ALPHABET) {
    const tile = document.createElement('button');
    tile.className = 'tile';
    tile.setAttribute('role', 'gridcell');
    tile.setAttribute('aria-label', `Burts ${ch}`);
    tile.textContent = ch;

    tile.addEventListener('click', () => {
      try { speak(ch); } catch { }
      highlightLetter(ch);
    });

    alphabetGrid.appendChild(tile);
  }
}

function highlightLetter(ch) {
  if (!alphabetGrid) return;
  if (highlighted === ch) return;
  highlighted = ch;
  alphabetGrid.querySelectorAll('.tile.active').forEach(el => el.classList.remove('active'));
  const tiles = Array.from(alphabetGrid.querySelectorAll('.tile'));
  const target = tiles.find(t => t.textContent === ch);
  if (target) {
    target.classList.add('active');
    target.focus({ preventScroll: true });
  }
}

// ----------------- Start -----------------
window.addEventListener('DOMContentLoaded', async () => {
  renderAlphabetGrid();
  await loadPretrainedModel();
  await camera.start();
});
