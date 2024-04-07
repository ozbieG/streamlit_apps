import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
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

def train_and_evaluate(X_train, y_train, X_test, y_test):
    # Define Logistic Regression model
    model = LogisticRegressionCV(Cs=10, cv=5, penalty='l2', max_iter=1000)
    # Train the model
    model.fit(X_train, y_train)
    # Test the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

# Streamlit App
def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("Predictive Maintenance Model")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Step 1: Exploratory Data Analysis (EDA)
        st.subheader("Step 1: Exploratory Data Analysis (EDA)")
        st.write("Distribution of the target variable (machine_status):")
        st.write(df['machine_status'].value_counts())

        if st.button("Next: Correlation Heatmap"):
            # Step 2: Correlation heatmap
            st.subheader("Step 2: Correlation Heatmap")
            df_numeric = df.drop(columns=['timestamp'])  # Drop non-numeric column
            plt.figure(figsize=(12, 8))
            sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', fmt=".2f")
            st.pyplot()

            if st.button("Next: Feature Selection and Preprocessing"):
                # Step 3: Preprocess data
                st.subheader("Step 3: Feature Selection and Preprocessing")
                X, y, selected_features = preprocess_data(df)

                # Allow manual editing of selected features
                st.write("Selected Features:")
                selected_features_editable = st.text_input("Manually edit selected features (comma-separated)", value=", ".join(selected_features))
                selected_features = [feature.strip() for feature in selected_features_editable.split(",")]

                if st.button("Next: Model Training and Evaluation"):
                    # Step 4: Train and evaluate model
                    st.subheader("Step 4: Model Training and Evaluation")

                    # Train-test split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                    accuracy = train_and_evaluate(X_train, y_train, X_test, y_test)
                    st.write("Average Accuracy:", accuracy)

if __name__ == "__main__":
    main()
