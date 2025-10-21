# backend/app/ml_handler.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import os
import tempfile
from collections import Counter

# --- 1. Configuration and Constants ---
WEATHER_CLASSES = ["Clear Day", "Impending Rain (Low Pressure)", "Cloudy/Overcast", "High Wind/Storm Warning", "Unknown/Ambiguous"]
NUM_CLASSES = len(WEATHER_CLASSES)
SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
N_MFCC = 39
MAX_TIME_FRAMES = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 2. Model Architectures ---
class AutoencoderClassifierPlaceholder(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AutoencoderClassifierPlaceholder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(N_MFCC, 128), nn.ReLU(), nn.Linear(128, 64))
        self.classifier = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, num_classes))
    def forward(self, x):
        encoded_features = self.encoder(x)
        pooled_features = torch.mean(encoded_features.transpose(1, 2), dim=2)
        x = self.classifier(pooled_features)
        return x

class CRNNPlaceholder(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CRNNPlaceholder, self).__init__()
        FINAL_CNN_CHANNELS = 96
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2, 1)), nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.cnn_final = nn.Sequential(
            nn.MaxPool2d((16, 1)),
            nn.Conv2d(64, FINAL_CNN_CHANNELS, kernel_size=1)
        )
        RNN_INPUT_SIZE = 384
        HIDDEN_SIZE = 256
        self.gru = nn.GRU(RNN_INPUT_SIZE, HIDDEN_SIZE, num_layers=1, batch_first=True, bidirectional=True)
        CLASSIFIER_INPUT = 512
        self.fc = nn.Linear(CLASSIFIER_INPUT, num_classes)
    def forward(self, x):
        x = self.cnn(x); x = self.cnn_final(x)
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(B, T, -1)
        rnn_out, _ = self.gru(x)
        x = torch.max(rnn_out, dim=1)[0]
        x = self.fc(x)
        return x

class DeepCNNPlaceholder(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(DeepCNNPlaceholder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(nn.Linear(512, num_classes))
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class ResNetPlaceholder(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(ResNetPlaceholder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.res_block1_conv, self.res_block1_downsample = self._make_res_block(64, 128)
        self.res_block2_conv, self.res_block2_downsample = self._make_res_block(128, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    def _make_res_block(self, in_channels, out_channels):
        downsample = None
        if in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        return layers, downsample
    def forward(self, x):
        x = self.conv1(x)
        identity = self.res_block1_downsample(x) if self.res_block1_downsample else x
        out = self.res_block1_conv(x); x = F.relu(out + identity)
        identity = self.res_block2_downsample(x) if self.res_block2_downsample else x
        out = self.res_block2_conv(x); x = F.relu(out + identity)
        x = self.avgpool(x); x = x.view(x.size(0), -1); x = self.fc(x)
        return x

class RecurrentPlaceholder(nn.Module):
    def __init__(self, rnn_type='LSTM', num_classes=NUM_CLASSES):
        super(RecurrentPlaceholder, self).__init__()
        self.rnn_type = rnn_type
        rnn_input_size = N_MFCC
        HIDDEN_SIZE = 128
        RNN_OUTPUT_DIM = HIDDEN_SIZE * 2
        if rnn_type == 'GRU-LSTM':
            self.gru = nn.GRU(rnn_input_size, HIDDEN_SIZE, num_layers=1, batch_first=True, bidirectional=True)
            self.lstm = nn.LSTM(RNN_OUTPUT_DIM, HIDDEN_SIZE, num_layers=1, batch_first=True, bidirectional=True)
            CLASSIFIER_INPUT = RNN_OUTPUT_DIM
        elif rnn_type == 'LSTM':
            self.lstm = nn.LSTM(rnn_input_size, HIDDEN_SIZE, num_layers=2, batch_first=True, bidirectional=True)
            CLASSIFIER_INPUT = RNN_OUTPUT_DIM
        else:
            self.lstm = nn.LSTM(rnn_input_size, HIDDEN_SIZE, num_layers=1, batch_first=True, bidirectional=True)
            CLASSIFIER_INPUT = RNN_OUTPUT_DIM
        self.fc = nn.Linear(CLASSIFIER_INPUT, num_classes)
    def forward(self, x):
        if self.rnn_type == 'GRU-LSTM':
            rnn_out1, _ = self.gru(x)
            rnn_out, _ = self.lstm(rnn_out1)
        else:
            rnn_out, _ = self.lstm(x)
        x = torch.mean(rnn_out, dim=1)
        x = self.fc(x)
        return x

# --- 3. Model Loading Logic ---
MODEL_FILES = {
    "Autoencoder": "final_autoencoder.pt", "GRU-LSTM": "final_grulstm.pt",
    "ResNet": "final_resnet.pt", "LSTM": "final_lstm.pt",
    "CRNN": "final_crnn.pt", "DeepCNN": "final_deepcnn.pt",
}
MODEL_CLASSES = {
    "Autoencoder": AutoencoderClassifierPlaceholder,
    "GRU-LSTM": lambda: RecurrentPlaceholder(rnn_type='GRU-LSTM'),
    "ResNet": ResNetPlaceholder, "LSTM": lambda: RecurrentPlaceholder(rnn_type='LSTM'),
    "CRNN": CRNNPlaceholder, "DeepCNN": DeepCNNPlaceholder,
}

def load_all_models():
    loaded_models = {}
    print(f"Loading models onto {DEVICE}...")
    for name, filename in MODEL_FILES.items():
        path = os.path.join(os.path.dirname(__file__), 'models', filename)
        if not os.path.exists(path):
            print(f"File {path} not found. Skipping {name}.")
            continue
        try:
            model = MODEL_CLASSES[name]()
            state_dict = torch.load(path, map_location=DEVICE)
            if 'model_state_dict' in state_dict: state_dict = state_dict['model_state_dict']
            model.load_state_dict(state_dict, strict=False)
            model.to(DEVICE).eval()
            loaded_models[name] = model
            print(f"Successfully loaded {name}")
        except Exception as e:
            print(f"--- ERROR LOADING {name} ({filename}): {e} ---")
    return loaded_models

# Load models once when the server starts
ALL_MODELS = load_all_models()


# --- 4. Audio Preprocessing Function ---
def process_audio(audio_path):
    try:
        y, loaded_sr = librosa.load(audio_path, sr=SR, mono=True)
    except Exception as e:
        raise IOError(f"Failed to load audio file: {e}")

    inputs = {}
    # MFCC Processing
    mfccs = librosa.feature.mfcc(y=y, sr=loaded_sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mfcc=N_MFCC)
    mfcc_seq = ((mfccs - mfccs.mean()) / (mfccs.std() + 1e-6)).T
    if mfcc_seq.shape[0] < MAX_TIME_FRAMES:
        padded_mfccs = np.vstack([mfcc_seq, np.zeros((MAX_TIME_FRAMES - mfcc_seq.shape[0], N_MFCC))])
    else:
        padded_mfccs = mfcc_seq[:MAX_TIME_FRAMES, :]
    inputs['mfcc'] = torch.from_numpy(padded_mfccs).float().unsqueeze(0).to(DEVICE)

    # Spectrogram Processing
    mel_spec = librosa.feature.melspectrogram(y=y, sr=loaded_sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    norm_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-6)
    if norm_spec.shape[1] < MAX_TIME_FRAMES:
        padded_spec = np.hstack([norm_spec, np.zeros((N_MELS, MAX_TIME_FRAMES - norm_spec.shape[1]))])
    else:
        padded_spec = norm_spec[:, :MAX_TIME_FRAMES]
    tensor_1ch = torch.from_numpy(padded_spec).float().unsqueeze(0)
    inputs['spectrogram'] = tensor_1ch.repeat(1, 3, 1, 1).to(DEVICE)
    
    return inputs


# --- 5. Prediction Function (Ensemble of all models) ---
def predict_weather(audio_bytes: bytes) -> str:
    if not ALL_MODELS:
        raise RuntimeError("No models were loaded successfully. Cannot make a prediction.")

    # Save audio bytes to a temporary file for librosa to read
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_file_path = tmp_file.name

    try:
        inputs = process_audio(tmp_file_path)
        all_predictions = []
        
        with torch.no_grad():
            for model_name, model in ALL_MODELS.items():
                # Select the correct input tensor based on the model type
                if model_name in ["Autoencoder", "GRU-LSTM", "LSTM"]:
                    input_tensor = inputs['mfcc']
                else: # ResNet, CRNN, DeepCNN
                    input_tensor = inputs['spectrogram']
                
                output = model(input_tensor)
                predicted_index = torch.argmax(output, dim=1).item()
                all_predictions.append(predicted_index)
        
        # Use majority vote for the final prediction
        if not all_predictions:
            return "Could not get a prediction."
            
        most_common_index = Counter(all_predictions).most_common(1)[0][0]
        final_prediction = WEATHER_CLASSES[most_common_index]

    finally:
        # Clean up the temporary file to avoid leaving files behind
        os.unlink(tmp_file_path)

    return final_prediction

# --- 6. NEW FUNCTIONS FOR MODEL SELECTION ---

def get_loaded_model_names():
    """Returns a list of names of the models that were loaded successfully."""
    return list(ALL_MODELS.keys())

def predict_with_single_model(audio_bytes: bytes, model_name: str) -> str:
    """Runs prediction using only one specified model."""
    model = ALL_MODELS.get(model_name)
    if not model:
        raise ValueError(f"Model '{model_name}' was not found or not loaded.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_file_path = tmp_file.name

    try:
        inputs = process_audio(tmp_file_path)
        
        with torch.no_grad():
            if model_name in ["Autoencoder", "GRU-LSTM", "LSTM"]:
                input_tensor = inputs['mfcc']
            else: # ResNet, CRNN, DeepCNN
                input_tensor = inputs['spectrogram']
            
            output = model(input_tensor)
            predicted_index = torch.argmax(output, dim=1).item()
            prediction = WEATHER_CLASSES[predicted_index]

    finally:
        os.unlink(tmp_file_path)

    return prediction