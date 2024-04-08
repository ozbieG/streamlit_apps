import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import class_weight
import streamlit as st
import pandas as pd
import time

# Function to perform resampling and feature selection
def preprocess_data(df, feature_selection_method, feature_selection_threshold):
    X = df.drop(columns=['machine_status', 'timestamp'])
    y = df['machine_status']

    if feature_selection_method == "Random Forest Importance":
        clf = RandomForestClassifier()
        clf.fit(X, y)
        feature_importances = clf.feature_importances_
        selected_features = X.columns[feature_importances >= feature_selection_threshold]
        X_selected = X[selected_features]
    elif feature_selection_method == "SVM Weight Coefficients":
        clf = SVC(kernel="linear")
        clf.fit(X, y)
        coef_abs = np.abs(clf.coef_[0])
        selected_features = X.columns[coef_abs >= feature_selection_threshold]
        X_selected = X[selected_features]
    else:
        raise ValueError("Invalid feature selection method. Choose either 'Random Forest Importance' or 'SVM Weight Coefficients'.")

    return X_selected, y, selected_features

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

    # Initialize current step as a session state
    current_step = st.session_state.get('current_step', 0)

    # Generate a counter for file uploader keys
    file_uploader_counter = st.session_state.get('file_uploader_counter', 0)

    # Cache the loaded DataFrame to avoid re-reading the CSV on every button click
    @st.cache(allow_output_mutation=True)
    def load_data(uploaded_file):
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                return df
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
                return None  # Indicate error

    # Step 1: Exploratory Data Analysis (EDA) & Correlation Heatmap
    if current_step == 0:
        # Upload CSV file
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key=f"file_uploader_{file_uploader_counter}")  # Unique key
        df = load_data(uploaded_file)  # Call the cached function to load data
        st.subheader("Exploratory Data Analysis (EDA)")
        if df is not None:
            st.write("Distribution of the target variable (machine_status):")
            st.write(df['machine_status'].value_counts())
            st.write("Correlation Heatmap:")
            non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
            df_numeric = df.drop(columns=non_numeric_columns)
            plt.figure(figsize=(12, 8))
            sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', fmt=".2f")
            st.pyplot()

            if st.button("Next: Feature Selection"):
                current_step = 1  # Update step after EDA

    # Step 2: Feature Selection
    if current_step == 1:
        st.subheader("Feature Selection")
        selected_feature_selection_method = st.selectbox("Select feature selection method", ["Random Forest Importance", "SVM Weight Coefficients"])

        # Select feature selection threshold
        feature_selection_threshold = st.slider("Select feature selection threshold", min_value=0.0, max_value=1.0, value=0.05, step=0.05)

        if st.button("Execute"):
            if df is not None:  # Check if df is not None before preprocessing
                X, y, selected_features = preprocess_data(df, selected_feature_selection_method, feature_selection_threshold)

                # Allow manual editing of selected features using checkbox list
                st.write("Selected Features:")
                selected_features_editable = st.multiselect("Select features to include", selected_features, default=selected_features.tolist())

                if st.button("Next: Model Training and Evaluation"):
                    current_step = 2  # Update step after Feature Selection

    # Step 3: Model Training and Evaluation
    if current_step == 2:
        st.subheader("Model Training and Evaluation")

        if df is not None:  # Check if df is not None before training and evaluation
            # Select the model
            selected_model = st.selectbox("Select model", ["Logistic Regression", "Random Forest Classifier", "Support Vector Machine (SVM)"])

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            accuracy = train_and_evaluate(X_train, y_train, X_test, y_test, selected_model)
            st.write("Average Accuracy:", accuracy)

    # Save current step and file uploader counter in session state
    st.session_state['current_step'] = current_step
    st.session_state['file_uploader_counter'] = file_uploader_counter + 1

if __name__ == "__main__":
    main()
