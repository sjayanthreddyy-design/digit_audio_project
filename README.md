# Digit Classification from Audio

Lightweight prototype that listens to spoken digits (0–9) and predicts the correct number.  
Built using the Free Spoken Digit Dataset (FSDD), MFCC feature extraction, and a simple logistic regression model.

## Features
- Uses **MFCC** features for compact audio representation.
- Trains a **Logistic Regression** classifier for fast and lightweight inference.
- Optional **live microphone prediction** for real-time testing.
- Minimal dependencies and easy to run.

## Dataset
This project uses the [Free Spoken Digit Dataset](https://huggingface.co/datasets/patrickvonplaten/spoken_digit) via the Hugging Face `datasets` library.

## Installation

pip install librosa soundfile scikit-learn torch torchaudio matplotlib datasets joblib sounddevice


## Training
Run the training script to load the dataset, extract MFCC features, train the model, and save it.

python train_model.py

This will print accuracy and confusion matrix, then save the trained model to `digit_model.pkl`.

## Live Prediction (Optional)
Run the live microphone test to speak digits and get predictions.

python predict_live.py

Speak clearly when prompted.  
Requires a working microphone.



## LLM Collaboration
Development was guided using LLM-based coding assistance for:
- Structuring a fast MFCC-based pipeline.
- Selecting a lightweight model for rapid inference.
- Integrating live microphone testing in minimal lines of code.
- Improving robustness against background noise.


