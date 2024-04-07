import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import class_weight
import streamlit as st
import pandas as pd

# Function to perform resampling and feature selection
def preprocess_data(df):
    X = df.drop(columns=['machine_status', 'timestamp'])
    y = df['machine_status']
    
    # Use SelectKBest with f_classif scoring function
    k_best = SelectKBest(score_func=f_classif, k=5)  # Select top 5 features
    X_selected = k_best.fit_transform(X, y)
    
    # Get selected feature indices
    selected_indices = k_best.get_support(indices=True)
    selected_features = X.columns[selected_indices]
    
    return X[selected_features], y, selected_features

def train_and_evaluate(X_train, y_train, X_test, y_test, model_name):
    if model_name == "Logistic Regression":
        model = LogisticRegressionCV(Cs=10, cv=5, penalty='l2', max_iter=1000)
    elif model_name == "Random Forest Classifier":
        model = RandomForestClassifier()
    elif model_name == "Support Vector Machine (SVM)":
        model = SVC()
    else:
        raise ValueError("Invalid model name. Choose either 'Logistic Regression', 'Random Forest Classifier', or 'Support Vector Machine (SVM)'.")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

# Streamlit App
def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("Predictive Maintenance Model")

    # Define step state
    steps = ["Step 1: Exploratory Data Analysis (EDA)",
             "Step 2: Correlation Heatmap",
             "Step 3: Feature Selection and Preprocessing",
             "Step 4: Model Training and Evaluation"]
    current_step = st.session_state.get("current_step", 0)

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Step 1: Exploratory Data Analysis (EDA)
        if current_step == 0:
            st.subheader(steps[0])
            st.write("Distribution of the target variable (machine_status):")
            st.write(df['machine_status'].value_counts())

            if st.button("Next: Correlation Heatmap"):
                st.session_state["current_step"] = current_step + 1

        # Step 2: Correlation heatmap
        elif current_step == 1:
            st.subheader(steps[1])
            df_numeric = df.drop(columns=['timestamp'])  # Drop non-numeric column
            plt.figure(figsize=(12, 8))
            sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', fmt=".2f")
            st.pyplot()

            if st.button("Next: Feature Selection and Preprocessing"):
                st.session_state["current_step"] = current_step + 1

        # Step 3: Feature Selection and Preprocessing
        elif current_step == 2:
            st.subheader(steps[2])
            X, y, selected_features = preprocess_data(df)

            # Allow manual editing of selected features using checkbox list
            st.write("Suggested Features:")
            selected_features_editable = st.multiselect("Select features to include", selected_features, default=selected_features.tolist())

            if st.button("Next: Model Training and Evaluation"):
                st.session_state["current_step"] = current_step + 1

        # Step 4: Model Training and Evaluation
        elif current_step == 3:
            st.subheader(steps[3])

            # Select the model
            selected_model = st.selectbox("Select model", ["Logistic Regression", "Random Forest Classifier", "Support Vector Machine (SVM)"])

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            accuracy = train_and_evaluate(X_train, y_train, X_test, y_test, selected_model)
            st.write("Average Accuracy:", accuracy)

if __name__ == "__main__":
    main()
