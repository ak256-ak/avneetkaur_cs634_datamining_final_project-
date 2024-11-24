import pandas as pd
import matplotlib.pyplot as plt

def summarize_results():
    
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

    return df  
def plot_metrics(df):
   
    metrics = ["Accuracy", "F1-Score", "ROC-AUC"]
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.bar(df["Model"], df[metric], color=["skyblue", "orange", "green"])
        plt.title(f"{metric} Comparison")
        plt.ylabel(metric)
        plt.xlabel("Model")
        plt.grid(axis="y")
        plot_path = f"outputs/{metric.lower()}_comparison.png"
        plt.savefig(plot_path)
        print(f"{metric} plot saved as {plot_path}")
        plt.show()

if __name__ == "__main__":
    df = summarize_results()  
    plot_metrics(df)  