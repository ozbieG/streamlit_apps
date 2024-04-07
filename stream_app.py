import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectFromModel
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import class_weight
import streamlit as st
import pandas as pd

# Function to perform resampling and feature selection
def preprocess_data(df):
    X = df.drop(columns=['machine_status', 'timestamp'])
    y = df['machine_status']

    # Perform resampling to address class imbalance
    resampling_pipeline = Pipeline([
        ('oversample', SMOTE(sampling_strategy=0.5, random_state=42)),
        ('undersample', RandomUnderSampler(sampling_strategy=1.0, random_state=42))
    ])

    X_resampled, y_resampled = resampling_pipeline.fit_resample(X, y)

    # Perform feature selection using Logistic Regression
    model = LogisticRegressionCV(Cs=10, cv=5, penalty='l2', max_iter=1000)
    model.fit(X_resampled, y_resampled)
    importance = np.abs(model.coef_[0])
    feature_names = X.columns

    # Select features with non-zero coefficients
    selected_features = feature_names[model.coef_[0] != 0]

    return X_resampled[selected_features], y_resampled, selected_features

# Function to train model and evaluate using Stratified KFold
def train_and_evaluate(X, y):
    # Define Stratified KFold for balanced splits
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Track accuracy scores across folds
    accuracies = []
    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Calculate class weights
        class_weights = class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)

        # Model Training and Evaluation within the loop (for each fold)
        model = LogisticRegressionCV(Cs=10, cv=5, penalty='l2', max_iter=1000, class_weight=dict(enumerate(class_weights)))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        accuracies.append(accuracy)

    # Return average accuracy across folds
    return np.mean(accuracies)

# Streamlit App
def main():
    st.title("Machine Learning Pipeline with Streamlit")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("Exploratory Data Analysis (EDA)")
        st.write("Distribution of the target variable (machine_status):")
        st.write(df['machine_status'].value_counts())

        # Correlation heatmap
        st.write("Correlation Heatmap")
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        st.pyplot()

        # Preprocess data
        st.subheader("Feature Selection and Preprocessing")
        X, y, selected_features = preprocess_data(df)
        st.write("Selected Features:")
        st.write(selected_features)

        # Train and evaluate model
        st.subheader("Model Training and Evaluation")
        accuracy = train_and_evaluate(X, y)
        st.write("Average Accuracy:", accuracy)

if __name__ == "__main__":
    main()
