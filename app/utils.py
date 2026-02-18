import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import tensorflow as tf
import timm
import sys
import os
import importlib.util
import types
from pathlib import Path
import tensorflow_hub as hub

# setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE_DIR / "weights"
MODEL_PY_PATH = Path(__file__).resolve().parent / "models.py"

# pytorch_utils safe import hack
pytorch_utils = types.ModuleType("pytorch_utils")
sys.modules["pytorch_utils"] = pytorch_utils

pytorch_utils.do_mixup = lambda *args, **kwargs: None
pytorch_utils.interpolate = lambda *args, **kwargs: None
pytorch_utils.pad_framewise_output = lambda *args, **kwargs: None

# force import of models.py to ensure all classes/functions are in the global namespace
if MODEL_PY_PATH.exists():
    spec = importlib.util.spec_from_file_location("models", str(MODEL_PY_PATH))
    models_module = importlib.util.module_from_spec(spec)
    sys.modules["models"] = models_module
    spec.loader.exec_module(models_module)
    print("Successfully forced 'models' namespace from models.py")
else:
    print(f"ERROR: Could not find {MODEL_PY_PATH}")


import timm.layers as layers
sys.modules['timm.models.layers.conv2d_same'] = layers
from timm.layers import Conv2dSame

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SIZE = 384

# LSTM classifier definition 
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=1280, hidden_size=256, num_classes=2):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

import __main__
setattr(__main__, "LSTMClassifier", LSTMClassifier)

# Load all models at startup
try:
# CNN
    cnn_model = torch.load(str(WEIGHTS_DIR / "cnn.pth"), map_location=DEVICE, weights_only=False)
    cnn_model.eval()

# LSTM
    lstm_model = torch.load(str(WEIGHTS_DIR / "lstm_classifier.pth"), map_location=DEVICE, weights_only=False)
    lstm_model.eval()

# PANNs
    panns_model = torch.load(str(WEIGHTS_DIR / "panns_feature_extractor.pth"), map_location=DEVICE, weights_only=False)
    panns_model.eval()
    
    print("All Torch models loaded successfully!")
except Exception as e:
    print(f"Loading failed: {e}")
BASE_DIR = Path(__file__).resolve().parent.parent
HUB_CACHE_DIR = BASE_DIR / "tfhub_cache"
HUB_CACHE_DIR.mkdir(exist_ok=True)

# Set the environment variable 
os.environ["TFHUB_CACHE_DIR"] = str(HUB_CACHE_DIR)
# Stress classifier
stress_classifier = tf.keras.models.load_model(str(WEIGHTS_DIR / "stress_classifier.keras"))
YAMNET_URL = "https://tfhub.dev/google/yamnet/1"

try:
    print("⏳ Loading YAMNet from TFHub (this may take a minute on first run)...")
    yamnet_extractor = hub.load(YAMNET_URL)
    print("YAMNet Extractor loaded.")
except Exception as e:
    print(f"YAMNet failed to load: {e}")
    yamnet_extractor = None 

# Load custom classifier
yamnet_classifier = tf.keras.models.load_model(str(WEIGHTS_DIR / "stress_classifier_yamnet.keras"))
def spec_to_cnn_embedding(logmel):
    img = (logmel - logmel.min()) / (logmel.max() - logmel.min() + 1e-9)
    img = torch.tensor(img, dtype=torch.float32)
    img = F.interpolate(
        img.unsqueeze(0).unsqueeze(0),
        size=(TARGET_SIZE, TARGET_SIZE),
        mode="bilinear",
        align_corners=False
    ).squeeze()
    img3 = img.repeat(3, 1, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = cnn_model(img3)
    return emb

def preprocess_panns(file_path):
    waveform, sr = librosa.load(file_path, sr=16000, mono=True)
    waveform = torch.tensor(waveform, dtype=torch.float32)
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=1)
    min_samples = 16000
    if waveform.shape[-1] < min_samples:
        waveform = F.pad(waveform, (0, min_samples - waveform.shape[-1]))
    return waveform.unsqueeze(0).to(DEVICE)

def predict_cnn_lstm(file_path):
    """Predict stress probability using CNN + LSTM."""
    waveform, sr = librosa.load(file_path, sr=16000)
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_fft=1024, hop_length=320, n_mels=64, fmin=50, fmax=14000
    )
    logmel = librosa.power_to_db(mel)
    emb = spec_to_cnn_embedding(logmel)
    
    with torch.no_grad():
        logits = lstm_model(emb)
# Convert raw logits to probabilities (0.0 to 1.0)
        probs = F.softmax(logits, dim=1)
    stress_prob = probs[0][1].item() 
    return stress_prob

def predict_panns(file_path):
    """Predict stress probability using PANNs + Keras classifier."""
    waveform = preprocess_panns(file_path)
    with torch.no_grad():
        embedding = panns_model(waveform)
    
    embedding_np = embedding["embedding"].cpu().numpy()
    preds = stress_classifier.predict(embedding_np, verbose=0)
    stress_prob = float(preds[0][1])
    return stress_prob

def predict_yamnet(file_path):
    """Predict stress level using YAMNet embeddings + Custom Classifier."""
# Load and preprocess
    waveform, sr = librosa.load(file_path, sr=16000, mono=True)
    waveform = waveform.astype(np.float32)
    
# Get embeddings from YAMNet
    _, embeddings, _ = yamnet_extractor(waveform)
    mean_embedding = tf.reduce_mean(embeddings, axis=0)
    mean_embedding = tf.expand_dims(mean_embedding, 0)
    preds = yamnet_classifier.predict(mean_embedding, verbose=0)
    stress_prob = float(preds[0][1])
    return stress_prob
