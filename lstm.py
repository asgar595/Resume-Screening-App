# lstm_resume_classifier.py

import numpy as np
import pandas as pd

# ------- 1. Load and prepare data -------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout

# ------------------------------
# STEP 1: Load your dataset
# ------------------------------
# CSV should have columns: "resume_text" and "label"
df = pd.read_csv("resumes.csv")   # <-- put your file name here

texts = df["resume_text"].astype(str).values      # X
labels = df["label"].astype(str).values           # y (job role / domain)

# ------------------------------
# STEP 2: Encode labels (string -> numbers)
# ------------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)        # e.g. "Data Scientist" -> 0, "Web Dev" -> 1 ...
num_classes = len(label_encoder.classes_)
y_cat = to_categorical(y_encoded, num_classes=num_classes)

# ------------------------------
# STEP 3: Tokenize and pad text
# ------------------------------
vocab_size = 20000        # max number of words to keep in vocab
max_len = 300             # max tokens per resume (truncate/pad)

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<UNK>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
X_padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

# ------------------------------
# STEP 4: Train–test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_padded, y_cat, test_size=0.2, random_state=42, stratify=y_cat
)

# ------------------------------
# STEP 5: Build Bi-LSTM model
# ------------------------------
embedding_dim = 128
lstm_units = 128

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
    Bidirectional(LSTM(lstm_units, return_sequences=False)),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print(model.summary())

# ------------------------------
# STEP 6: Train the model
# ------------------------------
history = model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=5,            # increase if you have time
    batch_size=32,
    verbose=1
)

# ------------------------------
# STEP 7: Evaluate
# ------------------------------
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc:.4f}")

# ------------------------------
# STEP 8: Use model for prediction
# ------------------------------
def predict_resume_category(raw_text):
    seq = tokenizer.texts_to_sequences([raw_text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    probs = model.predict(padded)
    class_id = np.argmax(probs, axis=1)[0]
    class_name = label_encoder.inverse_transform([class_id])[0]
    return class_name, probs[0][class_id]

# Example:
sample_resume = """
Experienced Python developer with 3+ years in data analysis, machine learning,
Pandas, NumPy, scikit-learn, and building ML APIs with Flask/FastAPI.
"""
pred_label, confidence = predict_resume_category(sample_resume)
print("Predicted Category:", pred_label)
print("Confidence:", confidence)
