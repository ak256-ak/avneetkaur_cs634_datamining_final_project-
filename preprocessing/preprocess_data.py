import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_path, output_folder):
    """
    Preprocess the dataset:
    - Handle missing or invalid values.
    - Normalize features using StandardScaler.
    - Split data into training and testing sets.
    - Save preprocessed datasets to CSV files.

    Args:
        input_path (str): Path to the input dataset (CSV file).
        output_folder (str): Folder to save preprocessed files.

    Returns:
        None
    """
    print("Starting preprocessing...")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    # Load the dataset
    df = pd.read_csv(input_path)
    print("Data loaded successfully")

    columns_to_impute = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in columns_to_impute:
        df[col].replace(0, pd.NA, inplace=True)
        df[col].fillna(df[col].median(), inplace=True)

    print("Missing values handled")

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the dataset into training and testing sets)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Data split into training and testing sets")

    # Sav preprocessed files
    pd.DataFrame(X_train).to_csv(f"{output_folder}/X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv(f"{output_folder}/X_test.csv", index=False)
    pd.DataFrame(y_train).to_csv(f"{output_folder}/y_train.csv", index=False)
    pd.DataFrame(y_test).to_csv(f"{output_folder}/y_test.csv", index=False)

    print("Preprocessed data saved successfully in", output_folder)

if __name__ == "__main__":
    preprocess_data(input_path="data/diabetes.csv", output_folder="data")
