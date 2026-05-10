# 🫁 PneumoScan AI — Chest X-Ray Pneumonia Detector

An AI-powered web application that detects pneumonia from chest X-ray images using deep learning (CNN).

---

## 📁 Project Structure

```
pneumonia-detector/
├── index.html          ← The web app (open this in browser)
├── style.css           ← Premium dark-themed styling
├── app.js              ← Application logic + TensorFlow.js inference
├── train_in_colab.py   ← Training script (run in Google Colab)
├── model/              ← Place your trained model here (after training)
│   ├── model.json
│   └── group1-shard1of1.bin
└── README.md           ← You are here
```

---

## 🚀 Quick Start (Demo Mode)

1. Open `index.html` in your browser
2. Upload any chest X-ray image
3. The app runs in **Demo Mode** with simulated predictions

---

## 🧠 Training Your Own Model (Google Colab)

### Step 1: Upload Dataset to Google Drive
1. Go to [Google Drive](https://drive.google.com/)
2. Upload your `pneumonia_dataset` folder (from Downloads)

### Step 2: Open Google Colab
1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a **New Notebook**
3. Change runtime to **GPU**: `Runtime` → `Change runtime type` → `T4 GPU`

### Step 3: Copy the Training Code
1. Open `train_in_colab.py` from this project
2. Copy each **CELL** section into separate Colab cells
3. Update `DATASET_PATH` to match your Google Drive path
4. Run all cells — training takes ~10-20 minutes with GPU

### Step 4: Download the Model
1. After training, the script converts the model to TensorFlow.js format
2. It will auto-download a `tfjs_model_download.zip` file
3. Extract it into the `model/` folder in this project

### Step 5: Run with Real Model
1. You need a local web server (browsers block local file loading)
2. Use one of these:
   - **VS Code**: Install "Live Server" extension, right-click `index.html` → Open with Live Server
   - **Python**: `python -m http.server 8000` then open `http://localhost:8000`
   - **Node.js**: `npx serve .` then open the URL shown

---

## 📊 Dataset Info

- **Source**: Kaggle Chest X-Ray dataset
- **Classes**: 
  - `0` = Normal (healthy lungs)
  - `1` = Bacterial Pneumonia
  - `2` = Viral Pneumonia
- **Total Training Images**: 4,672
- **Image Format**: JPG

---

## ⚠️ Disclaimer

This project is for **educational and demonstration purposes only**.
It is NOT a medical device and should NOT be used for actual medical diagnosis.
Always consult a qualified healthcare professional for medical advice.
