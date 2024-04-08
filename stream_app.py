import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import class_weight
import streamlit as st
import pandas as pd
import time
from sklearn.utils import resample

# Function to perform resampling and feature selection
@st.cache(allow_output_mutation=True)
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

    yes_indices = y[y == 1].index
    no_indices = y[y == 0].index

    downsampled_yes_indices = resample(yes_indices, replace=False, n_samples=int(0.1 * len(yes_indices)), random_state=42)
    new_indices = np.concatenate([downsampled_yes_indices, no_indices])
    X_selected = X_selected.loc[new_indices]
    y = y[new_indices]
    return X_selected, y, selected_features



def train_and_evaluate(X_train, y_train, X_test, y_test, model_name):
    if model_name == "Logistic Regression":
        model = LogisticRegressionCV(Cs=10, cv=5, penalty='l2', max_iter=5000)
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

    # Initialize session state variables
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'file_uploader_counter' not in st.session_state:
        st.session_state.file_uploader_counter = 0
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'selected_features' not in st.session_state:
        st.session_state.selected_features = []
    if 'X' not in st.session_state:
        st.session_state.X = []
    if 'y' not in st.session_state:
        st.session_state.y = []
    if 'button_click' not in st.session_state:
        st.session_state.button_click = False
    if 'button_click1' not in st.session_state:
        st.session_state.button_click1 = False
    if 'selected_feature_selection_method' not in st.session_state:
        st.session_state.selected_feature_selection_method = "Random Forest Importance"
    if 'feature_selection_threshold' not in st.session_state:
        st.session_state.feature_selection_threshold = 0.05
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "Logistic Regression"

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
    if st.session_state.df is None:
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key=f"file_uploader_{st.session_state.file_uploader_counter}")
        if uploaded_file is not None:
            st.session_state.df = load_data(uploaded_file)
            st.session_state.file_uploader_counter += 1
    
    if st.session_state.df is not None:
        st.subheader("Exploratory Data Analysis (EDA)")
        st.write("Distribution of the target variable (machine_status):")
        st.write(st.session_state.df['machine_status'].value_counts())
        st.write("Correlation Heatmap:")
        non_numeric_columns = st.session_state.df.select_dtypes(exclude=[np.number]).columns.tolist()
        df_numeric = st.session_state.df.drop(columns=non_numeric_columns)
        plt.figure(figsize=(12, 8))
        sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        st.pyplot()
        
        # Step 2: Feature Selection
        st.subheader("Feature Selection")
        st.session_state.selected_feature_selection_method = st.selectbox("Select feature selection method", ["Random Forest Importance", "SVM Weight Coefficients"])

        # Select feature selection threshold
        st.session_state.feature_selection_threshold = st.slider("Select feature selection threshold", min_value=0.0, max_value=1.0, value=0.05, step=0.05)

        if st.button("Next"):
            st.session_state.button_click = True

    if st.session_state.button_click:
        X, y, selected_features = preprocess_data(st.session_state.df, st.session_state.selected_feature_selection_method, st.session_state.feature_selection_threshold)
        st.session_state.selected_features = selected_features
        st.session_state.X = X
        st.session_state.y = y
        # Allow manual editing of selected features using checkbox list
        st.write("Selected Features:")
        selected_features_editable = st.multiselect("Select features to include", selected_features, default=list(st.session_state.selected_features))

        # Step 3: Model Training and Evaluation
        st.subheader("Model Training and Evaluation")

        # Select the model
        st.session_state.selected_model = st.selectbox("Select model", ["Logistic Regression", "Random Forest Classifier", "Support Vector Machine (SVM)"])
        if st.button("Train and Test"):
            st.session_state.button_click1 = True
        # Train-test split
    if st.session_state.button_click1:
        X_train, X_test, y_train, y_test = train_test_split(st.session_state.X, st.session_state.y, test_size=0.3, random_state=42)
        accuracy = train_and_evaluate(X_train, y_train, X_test, y_test, st.session_state.selected_model)
        st.write("Average Accuracy:", accuracy)

if __name__ == "__main__":
    main()
