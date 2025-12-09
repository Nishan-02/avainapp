# 🦅 Avian Weather Net

### A Deep Learning Ensemble for Weather Prediction Using Bioacoustics

[![Live Demo](https://img.shields.io/badge/Live_Demo-Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://avainapp.vercel.app)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)

## 📖 Project Overview

**Avian Weather Net** is a research-driven Deep Learning project that bridges the gap between bioacoustics and meteorology. Our central research hypothesis investigates whether bird vocalizations—their calls, songs, and patterns—can serve as reliable, real-time indicators of local weather conditions.

To test this, we engineered a sophisticated data pipeline and a voting ensemble of **6 Deep Learning Models**. The system processes raw audio recordings (MP3/WAV) to classify the environment into five specific meteorological categories.

**Target Classes:** `Sunny` | `Rainy` | `Cloudy` | `Windy` | `Foggy`

---

## 🧠 Deep Learning Architecture

The core intelligence of this project lies in its **Multi-Model Ensemble**. We trained distinct architectures to capture different aspects of audio data (spatial features via Spectrograms and temporal sequences via MFCCs).

| Model | Architecture Type | Role & Specialty |
| :--- | :--- | :--- |
| **DeepCNN** | Deep Convolutional NN | Extracts spatial features from Mel-Spectrogram images. |
| **ResNet** | Residual Network | Prevents vanishing gradients; captures deep hierarchical features. |
| **CRNN** | Conv. Recurrent NN | A hybrid model capturing both spatial (Conv2D) and temporal (RNN) dependencies. |
| **LSTM** | Long Short-Term Memory | Specialized in analyzing long-term dependencies in time-series audio data. |
| **GRU-LSTM** | Gated Recurrent Unit + LSTM | Optimized for sequence learning with lower computational cost than pure LSTM. |
| **Autoencoder** | Unsupervised Learning | Used for dimensionality reduction and latent feature extraction. |

**Inference Mechanism:** The backend aggregates the softmax probability outputs from all six models and applies a **Majority Vote** algorithm to determine the final prediction with high confidence.

---

## 🛠️ Tech Stack

### Data Science & DL
* **Frameworks:** PyTorch, TorchAudio
* **Audio Processing:** Librosa (MFCC & Mel-Spectrogram generation)
* **Data Handling:** NumPy, Pandas

### Backend Engineering
* **Framework:** FastAPI (High-performance Async support)
* **Server:** Uvicorn
* **Deployment:** Render

### Frontend Interface
* **Framework:** React.js
* **Styling:** Custom CSS (Theme: *Wet Asphalt & Butter*)
* **Deployment:** Vercel

---

## 📸 Screenshots

*(Add screenshots of your UI here later)*

---

## 🚀 Local Installation & Setup

Follow these steps to run the research project locally.

### 1. Clone the Repository
```bash
git clone [https://github.com/Nishan-02/avainapp.git](https://github.com/Nishan-02/avainapp.git)
cd avainapp
cd backend

# Create virtual environment
python -m venv .venv

# Activate environment
# On Windows:
.\.venv\Scripts\Activate
# On Mac/Linux:
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Run the API Server
uvicorn app.main:app --reload
# Open a new terminal
cd frontend

# Install Node modules
npm install

# Start the React App
npm start
