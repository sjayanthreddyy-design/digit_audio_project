import librosa
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

def extract_features(signal, sr):
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

data = load_dataset("patrickvonplaten/spoken_digit", split="train")
X, y = [], []

for item in data:
    feat = extract_features(item["audio"]["array"], item["audio"]["sampling_rate"])
    X.append(feat)
    y.append(item["label"])

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print(confusion_matrix(y_test, preds))

joblib.dump(model, "digit_model.pkl")
