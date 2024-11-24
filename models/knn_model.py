import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import os

def train_knn():
    """
    Train a K-Nearest Neighbors (KNN) model using preprocessed data and evaluate its performance.
    """
    X_train = pd.read_csv("data/X_train.csv")
    X_test = pd.read_csv("data/X_test.csv")
    y_train = pd.read_csv("data/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/y_test.csv").values.ravel()

    knn = KNeighborsClassifier(n_neighbors=5)

    knn.fit(X_train, y_train)
    print("KNN model trained successfully.")

    y_pred = knn.predict(X_test)
    y_prob = knn.predict_proba(X_test)[:, 1]

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
    plt.title("ROC Curve - KNN")
    plt.legend(loc="lower right")
    plt.grid()

    output_folder = "outputs"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    plot_path = os.path.join(output_folder, "roc_curve_knn.png")
    plt.savefig(plot_path)
    print(f"ROC curve saved as {plot_path}")
    plt.show()

if __name__ == "__main__":
    train_knn()
