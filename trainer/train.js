// ----------------- Globālie elementi -----------------
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

const labelSelect = document.getElementById('label');
const letterDisplay = document.getElementById('letter');
const countsDisplay = document.getElementById('counts');
const mFps = document.getElementById('mFps');
const mLat = document.getElementById('mLat');
const mPred = document.getElementById('mPred');
const minConfInput = document.getElementById('minConf');
const minConfVal = document.getElementById('minConfVal');

const btnSave = document.getElementById('btnSave');
const btnTrain = document.getElementById('btnTrain');
const btnExport = document.getElementById('btnExport');
const btnClear = document.getElementById('btnClear');

// ----------------- Stāvoklis / veikals -----------------
const store = { MIN_CONFIDENCE: Number(minConfInput.value) };
minConfVal.textContent = store.MIN_CONFIDENCE.toFixed(2);
minConfInput.addEventListener('input', () => {
  store.MIN_CONFIDENCE = Number(minConfInput.value);
  minConfVal.textContent = store.MIN_CONFIDENCE.toFixed(2);
});

// ----------------- Datu kolekcija un modelis -----------------
const samples = [];   // [ [x1,y1,z1, ...], ... ]
const labels = [];    // ["A","B",...]
let trainedModel = null;
let labelSet = [];    // apmācīto klašu unikālais saraksts
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

function updateCounts() {
  const counts = {};
  labels.forEach(l => counts[l] = (counts[l] || 0) + 1);
  const summary = Object.entries(counts)
    .sort((a, b) => a[0].localeCompare(b[0]))
    .map(([k, v]) => `${k}: ${v}`).join('\n');
  countsDisplay.textContent = summary || 'Nav saglabātu žestu.';
}

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

// ---- Persistences palīgi ----
function persistSamples() {
  try {
    localStorage.setItem("gestureSamples", JSON.stringify(samples));
    localStorage.setItem("gestureLabels", JSON.stringify(labels));
  } catch (e) { console.warn("Persist samples failed:", e); }
}

async function saveModelAllWays() {
  try { await trainedModel.save("indexeddb://gestureModel"); }
  catch (e) { console.warn("IndexedDB save failed:", e); }

  try { await trainedModel.save("localstorage://gestureModel"); }
  catch (e) { console.warn("LocalStorage model save failed:", e); }
}

async function tryLoadAnySavedModel() {
  try {
    const list = await tf.io.listModels();
    if (list["indexeddb://gestureModel"]) return await tf.loadLayersModel("indexeddb://gestureModel");
    if (list["localstorage://gestureModel"]) return await tf.loadLayersModel("localstorage://gestureModel");
  } catch (e) {
    console.warn("Model auto-load failed:", e);
  }
  return null;
}

// ----------------- Datu darbības -----------------
function saveSample() {
  if (!latestLandmarks) return alert("Nav rokas datu!");
  const flat = flattenLandmarks(latestLandmarks);
  samples.push(flat);
  labels.push(labelSelect.value);
  updateCounts();
  updateTrainingStateOnGrid();
  persistSamples();
}

async function trainModel() {
  if (samples.length < 5) { alert("Nepietiek paraugu (≥5)."); return; }

  const xs = tf.tensor2d(samples);
  labelSet = Array.from(new Set(labels));
  const idxs = labels.map(l => labelSet.indexOf(l));
  const ys = tf.oneHot(tf.tensor1d(idxs, 'int32'), labelSet.length);

  trainedModel = tf.sequential();
  trainedModel.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [xs.shape[1]] }));
  trainedModel.add(tf.layers.dropout({ rate: 0.15 }));
  trainedModel.add(tf.layers.dense({ units: 32, activation: 'relu' }));
  trainedModel.add(tf.layers.dropout({ rate: 0.10 }));
  trainedModel.add(tf.layers.dense({ units: labelSet.length, activation: 'softmax' }));
  trainedModel.compile({ optimizer: tf.train.adam(0.0015), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

  await trainedModel.fit(xs, ys, {
    epochs: 25,
    batchSize: Math.min(32, samples.length),
    shuffle: true
  });

  smoothScores = new Array(labelSet.length).fill(0);

  // ✅ metadata ar klašu secību publiskajai versijai
  trainedModel.metadata = { labelSet };

  persistSamples();
  await saveModelAllWays();

  alert("Modelis apmācīts un saglabāts lokāli! Vari eksportēt uz GitHub.");
  btnExport.disabled = false;

  xs.dispose(); ys.dispose();
  updateTrainingStateOnGrid();
}

async function exportModelFiles() {
  if (!trainedModel) return alert("Nav modeļa ko eksportēt.");

  await trainedModel.save(tf.io.withSaveHandler(async (artifacts) => {
    const modelJson = {
      modelTopology: artifacts.modelTopology,
      weightSpecs: artifacts.weightSpecs,
      metadata: trainedModel.metadata || {}
    };

    const jsonBlob = new Blob([JSON.stringify(modelJson)], { type: "application/json" });
    const weightsBlob = new Blob([artifacts.weightData], { type: "application/octet-stream" });

    const a1 = document.createElement("a");
    a1.href = URL.createObjectURL(jsonBlob);
    a1.download = "model.json";
    a1.click();

    const a2 = document.createElement("a");
    a2.href = URL.createObjectURL(weightsBlob);
    a2.download = "weights.bin";
    a2.click();

    return {
      modelArtifactsInfo: {
        dateSaved: new Date(),
        modelTopologyType: 'JSON',
        weightDataBytes: artifacts.weightData.byteLength
      }
    };
  }));

  alert("Modelis eksportēts! Ieliec model.json + weights.bin public/model/");
}

async function loadFromLocal() {
  try {
    // Paraugi
    const s = localStorage.getItem("gestureSamples");
    const l = localStorage.getItem("gestureLabels");
    if (s && l) {
      const sArr = JSON.parse(s); const lArr = JSON.parse(l);
      if (Array.isArray(sArr) && Array.isArray(lArr) && sArr.length === lArr.length) {
        samples.length = 0; labels.length = 0;
        samples.push(...sArr); labels.push(...lArr);
        updateCounts();
      }
    }

    // Modelis
    const mdl = await tryLoadAnySavedModel();
    if (mdl) {
      trainedModel = mdl;
      // ja metadata ir saglabājies, izvelkam labelSet
      const meta = trainedModel.metadata || {};
      labelSet = meta.labelSet || Array.from(new Set(labels));
      smoothScores = new Array(labelSet.length).fill(0);
      btnExport.disabled = false;
    }

    updateTrainingStateOnGrid();
  } catch (e) {
    console.warn("Ielāde izgāzusies:", e);
  }
}

function clearAll() {
  if (!confirm("Dzēst visus lokālos paraugus un modeli?")) return;
  samples.length = 0; labels.length = 0;
  localStorage.removeItem("gestureSamples");
  localStorage.removeItem("gestureLabels");
  tf.io.removeModel("indexeddb://gestureModel").catch(() => {});
  tf.io.removeModel("localstorage://gestureModel").catch(() => {});
  trainedModel = null; labelSet = []; smoothScores = null;
  updateCounts();
  updateTrainingStateOnGrid();
  btnExport.disabled = true;
  alert("Notīrīts.");
}

function getLabelSet() { return Array.from(new Set(labelSet.length ? labelSet : labels)); }

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
    for (let i = 1; i < norm.length; i++) { if (norm[i] > maxV) { maxV = norm[i]; maxI = i; } }
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
  "A", "Ā", "B", "C", "Č", "D", "E", "Ē", "F", "G", "Ģ", "H",
  "I", "Ī", "J", "K", "Ķ", "L", "Ļ", "M", "N", "Ņ", "O", "P",
  "R", "S", "Š", "T", "U", "Ū", "V", "Z", "Ž"
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
      labelSelect.value = ch;
      try { speak(ch); } catch { }
      highlightLetter(ch);
    });

    alphabetGrid.appendChild(tile);
  }
  updateTrainingStateOnGrid();
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

function updateTrainingStateOnGrid() {
  if (!alphabetGrid) return;
  const trained = new Set(getLabelSet());
  alphabetGrid.querySelectorAll('.tile').forEach(tile => {
    const ch = tile.textContent;
    if (trained.size && !trained.has(ch)) tile.classList.add('muted');
    else tile.classList.remove('muted');
  });
}

// ----------------- UI notikumi + īsceļi -----------------
btnSave.onclick = () => saveSample();
btnTrain.onclick = () => trainModel();
btnExport.onclick = () => exportModelFiles();
btnClear.onclick = () => clearAll();

window.addEventListener('keydown', (e) => {
  const k = e.key.toLowerCase();
  if (k === 's') saveSample();
  if (k === 't') trainModel();
});

// Pirms lapas aizvēršanas – persistē paraugus
window.addEventListener('beforeunload', () => {
  try { persistSamples(); } catch {}
});

// ----------------- Start -----------------
window.addEventListener('DOMContentLoaded', async () => {
  try {
    renderAlphabetGrid();
    await loadFromLocal();
  } finally {
    updateCounts();
    updateTrainingStateOnGrid();
    await camera.start();
  }
});
