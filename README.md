# RokaRunā — Latviešu zīmju valodas alfabēta atpazinējs

RokaRunā is a real-time Latvian Sign Language (LZV) alphabet gesture recognizer.
It detects hand poses from the camera stream and predicts which letter of the Latvian alphabet you are signing.

The project includes two environments:

1. **Training Environment** (`trainer/`)  
   – collect gesture samples, train a model, export `model.json` + `weights.bin`.

2. **Public Runtime Environment** (`docs/`)  
   – loads the trained model and recognizes letters in real time (used with GitHub Pages).

The goal of this project is to make Latvian sign language more accessible, easier to learn, and fun to explore.

---

## 🎓 Training Environment (`trainer/`)

The training environment lets you collect gesture samples and train your own model directly in the browser.

---

### 🔧 How to open

Simply open the training interface locally in your browser: trainer/train.html

---

### 🧪 Training Steps

1. **Allow camera access** when the browser requests it.  
2. **Choose a letter** from the dropdown (e.g. `A`, `B`, `Č`, `Ņ`, etc.).  
3. Hold the correct sign for that letter in front of the camera.  
4. Click **“Save gesture”** multiple times to collect samples.  
   - Recommended: **20–50 samples per letter**, with slight variations in angle and position.  
5. Repeat the process for every letter you want the model to recognize.  
6. Click **“Train model”** and wait for training to finish.  
7. Once trained, click **“Export model”** — your browser will download: model.json, weights.bin

---

## 🌍 Public Runtime (`docs/`)

The public runtime is the version intended for regular users.  
It does **not** include training controls — only hand tracking and letter recognition.

---

### 📦 Model Placement

After exporting your trained model, copy the files into:
docs/model/model.json
docs/model/weights.bin


Make sure that **both filenames are exactly correct**, otherwise TensorFlow.js will not load the model.

---

### ▶️ How to Run Locally

Simply open the public interface: docs/index.html

---

## 🤝 Author

**Emīlija Ērgle**

RokaRunā is created as a project at the intersection of:

- accessibility and inclusion,

- sign language and technology,

- practical experimentation with MediaPipe & TensorFlow.js in the browser.

---

## 📜 License

This project is licensed under the MIT License.
You are free to use, modify and distribute it, as long as you keep the license notice.