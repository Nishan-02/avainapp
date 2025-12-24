# 🦅 Avian Weather Net  
### A Deep Learning Ensemble for Weather Prediction Using Bioacoustics

[![Live Demo](https://img.shields.io/badge/Live_Demo-Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://avainapp.vercel.app)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)

---

## 📖 Project Overview

**Avian Weather Net** is a research-driven Deep Learning project that explores the intersection of **bioacoustics and meteorology**.  
The core objective of this project is to determine whether **bird vocalizations**—including calls, songs, and acoustic patterns—can be used as reliable indicators of **local weather conditions**.

To achieve this, we designed an end-to-end system that processes raw bird audio recordings (MP3/WAV), extracts meaningful acoustic features, and classifies the environment into predefined weather categories using an **ensemble of deep learning models**.

### 🎯 Target Weather Classes
`Sunny` | `Rainy` | `Cloudy` | `Windy` | `Foggy`

---

## 🧠 Deep Learning Architecture

The intelligence of Avian Weather Net is built on a **Multi-Model Ensemble Strategy**.  
Each model is trained to capture different acoustic characteristics from bird audio, such as spatial patterns from spectrograms and temporal dependencies from MFCC sequences.

### 🔹 Ensemble Models Used

| Model | Architecture Type | Role & Specialty |
|------|-------------------|------------------|
| **DeepCNN** | Deep Convolutional Neural Network | Extracts spatial patterns from Mel-Spectrogram images. |
| **ResNet** | Residual Neural Network | Enables deep feature learning while preventing vanishing gradients. |
| **CRNN** | Convolutional Recurrent Neural Network | Captures both spatial (CNN) and temporal (RNN) audio features. |
| **LSTM** | Long Short-Term Memory Network | Learns long-term temporal dependencies in audio signals. |
| **GRU-LSTM** | GRU + LSTM Hybrid | Efficient sequence learning with reduced computational cost. |
| **Autoencoder** | Unsupervised Learning Model | Performs dimensionality reduction and latent feature extraction. |

### 🔁 Inference Strategy
During inference, probability outputs from all six models are combined using a **Majority Voting mechanism**, resulting in a more **robust and accurate final prediction**.

---

## 🎧 Audio Processing Pipeline

1. Raw bird audio collected from online sources and field recordings  
2. Noise handling and segmentation  
3. Feature extraction using:
   - Mel-Spectrograms
   - MFCCs
4. Feature normalization and batching  
5. Model-wise prediction and ensemble aggregation  

---

## 🛠️ Tech Stack

### 🔬 Data Science & Deep Learning
- **Frameworks:** PyTorch, TorchAudio  
- **Audio Processing:** Librosa (MFCC & Mel-Spectrograms)  
- **Data Handling:** NumPy, Pandas  

### ⚙️ Backend Engineering
- **Framework:** FastAPI  
- **Server:** Uvicorn  
- **Deployment:** Render  

### 🌐 Frontend Interface
- **Framework:** React.js  
- **Styling:** Custom CSS (Theme: *Wet Asphalt & Butter*)  
- **Deployment:** Vercel  

---

## 🚀 Features

- 🎶 Weather prediction from raw bird audio
- 🤖 Ensemble of 6 deep learning models
- 🌦️ Classification into 5 weather categories
- ⚡ FastAPI-based high-performance backend
- 🌐 Interactive web interface for real-time prediction

---

## 📌 Project Status

✅ Completed and deployed  
🌍 Live Web Application available  

---

## 🔮 Future Enhancements

- Increase dataset size with region-specific bird species
- Add attention-based deep learning models
- Real-time audio streaming support
- Mobile application integration
- Explainable AI (XAI) for model decisions

---

## ⭐ Acknowledgements

- Open-source bioacoustics and deep learning communities  
- Dataset contributors and research references  
- Team collaboration and academic guidance  
