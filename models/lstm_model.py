import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

def train_lstm():
    """
    Train an LSTM model using preprocessed data and evaluate its performance.
    """
    X_train = pd.read_csv("data/X_train.csv").values
    X_test = pd.read_csv("data/X_test.csv").values
    y_train = pd.read_csv("data/y_train.csv").values
    y_test = pd.read_csv("data/y_test.csv").values

    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)

    
    model = Sequential([
        LSTM(64, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(32, activation='tanh'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    print("LSTM model compiled successfully.")

    
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    print("LSTM model trained successfully.")

    
    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob > 0.5).astype(int)

   
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {roc_auc:.2f}")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})", color="darkorange")
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - LSTM")
    plt.legend(loc="lower right")
    plt.grid()

    
    output_folder = "outputs"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    
    plot_path = os.path.join(output_folder, "roc_curve_lstm.png")
    plt.savefig(plot_path)
    print(f"ROC curve saved as {plot_path}")
    plt.show()

if __name__ == "__main__":
    train_lstm()
