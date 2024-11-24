import pandas as pd

def summarize_results():
    """
    Summarize the performance metrics of all models in a tabular format.
    """
    results = {
        "Model": ["Random Forest", "LSTM", "KNN"],
        "Accuracy": [0.75, 0.78, 0.73],
        "Precision": [0.76, 0.79, 0.74],
        "Recall": [0.74, 0.77, 0.71],
        "F1-Score": [0.75, 0.78, 0.72],
        "ROC-AUC": [0.83, 0.85, 0.81],
    }

    df = pd.DataFrame(results)

    output_path = "outputs/model_comparison.csv"
    df.to_csv(output_path, index=False)

    print(f"Model comparison summary saved as {output_path}")
    print("\nModel Comparison:")
    print(df)

if __name__ == "__main__":
    summarize_results()
