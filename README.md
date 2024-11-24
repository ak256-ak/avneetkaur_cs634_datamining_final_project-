# avneetkaur_cs634_datamining_final_project-


# Overview

This project implements and compares three machine learning models

Random Forest ,LSTM, KNN to predict diabetes using a synthetic dataset. The models are evaluated based on their performance metrics, and visualizations are generated to aid in comparison.

Dataset:

The dataset contains the following features::


1. Pregnancies: Number of times pregnant.

2. Glucose: Plasma glucose concentration.

3. BloodPressure: Diastolic blood pressure (mm Hg).

4. SkinThickness: Triceps skinfold thickness (mm).

5. Insulin: 2-hour serum insulin (mu U/ml).

6. BMI: Body mass index (weight in kg/(height in m)^2).

7. DiabetesPedigreeFunction: Diabetes pedigree function score.

8. Age: Age of the patient.

(Outcome: Target variable (1 = diabetes, 0 = no diabetes).)

The dataset is preprocessed to handle missing or invalid values, normalize features, and split into training and testing sets.



# Models
The following models were implemented and evaluated:
1. Random Forest: An ensemble learning method using multiple decision trees.
2. LSTM: A type of recurrent neural network (RNN) suitable for sequence data.
3. KNN: An instance-based learning algorithm that classifies data based on proximity to neighbors.



# Results
The table below summarizes the performace 

| Model           | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-----------------|----------|-----------|--------|----------|---------|
| Random Forest   | 0.75     | 0.76      | 0.74   | 0.75     | 0.83    |
| LSTM            | 0.78     | 0.79      | 0.77   | 0.78     | 0.85    |
| KNN             | 0.73     | 0.74      | 0.71   | 0.72     | 0.81    


The following visualizations are included in the outputs folder:
accuracy_comparison.png
f1-score_comparison.png
roc-auc_comparison.png





# Clone the Repository

git clone 
cd avneetkaur_cs634_datamining_final_project

# install
pip3 install -r requirements.txt


# Preprocess the Data

python3 preprocessing/preprocess_data.py


# Train and Evaluate Models


Random Forest:

python3 models/random_forest.py

LSTM:

python3 models/lstm_model.py

KNN:

python3 models/knn_model.py

# Comparing Results


python3 evaluation/evaluate_models.py


# Install the required dependencies 

pip3 install -r requirements.txt
