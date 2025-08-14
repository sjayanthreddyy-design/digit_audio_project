import sounddevice as sd
import librosa
import numpy as np
import joblib

model = joblib.load("digit_model.pkl")

def extract_features(signal, sr):
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

while True:
    print("Speak a digit: ")
    audio = sd.rec(int(2 * 8000), samplerate=8000, channels=1)
    sd.wait()
    signal = audio[:, 0]
    feat = extract_features(signal, 8000).reshape(1, -1)
    pred = model.predict(feat)
    print("Predicted Digit:", pred[0])
