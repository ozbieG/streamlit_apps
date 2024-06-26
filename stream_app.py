import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV,LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import pandas as pd
import time
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix


# Function to perform resampling and feature selection
@st.cache(allow_output_mutation=True)
def preprocess_data(df, feature_selection_method, feature_selection_threshold):
    df = df.drop(columns=['timestamp'])
    X = df.drop(columns=['machine_status'])
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
    elif feature_selection_method == "Logistic Regression Coefficients":
        clf = LogisticRegression()
        clf.fit(X, y)
        coef_abs = np.abs(clf.coef_[0])
        selected_features = X.columns[coef_abs >= feature_selection_threshold]
        X_selected = X[selected_features]
    else:
        raise ValueError("Invalid feature selection method. Choose either 'Random Forest Importance' or 'SVM Weight Coefficients'.")

    yes_indices = y[y == 1].index
    no_indices = y[y == 0].index

    downsampled_yes_indices = resample(yes_indices, replace=False, n_samples=len(no_indices), random_state=42)
    new_indices = np.concatenate([downsampled_yes_indices, no_indices])
    X_whole = X_selected.copy()
    X_selected = X_selected.loc[new_indices]
    y = y[new_indices]

    return X_selected, y, selected_features,X_whole


def train_and_evaluate(X_train, y_train, X_test, y_test, model_name,X_whole):
    if model_name == "Logistic Regression":
        model = LogisticRegressionCV(Cs=10, cv=5, penalty='l2', max_iter=15000)
    elif model_name == "Random Forest Classifier":
        model = RandomForestClassifier(n_estimators=5, 
                               max_depth=3, 
                               min_samples_split=5, 
                               min_samples_leaf=3, 
                               max_features='sqrt', 
                               bootstrap=False)
    elif model_name == "Support Vector Machine (SVM)":
        model = SVC(probability=True)
    else:
        raise ValueError("Invalid model name. Choose either 'Logistic Regression', 'Random Forest Classifier', or 'Support Vector Machine (SVM)'.")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    probabilities = model.predict_proba(X_whole)
    total_pred = model.predict(X_whole)
    accuracy = accuracy_score(y_test, y_pred)
    return y_pred,y_test,accuracy,probabilities,total_pred

@st.cache(allow_output_mutation=True) 
def get_confusion_matrix(y_test,y_pred):
    return confusion_matrix(y_test, y_pred)

@st.cache(allow_output_mutation=True) 
def get_corr(df):
    return df.corr()

@st.cache(allow_output_mutation=True) 
def generate_recommended_days(probabilities, df):
    scaled_probabilities = probabilities * 100
    negative_class_probabilities = scaled_probabilities[:, 1]
    recommended_days = np.round(np.interp(negative_class_probabilities, [0,95, 96, 97, 98, 99, 100], [0, 5, 10, 15, 20, 25, 30])).astype(int)
    recommended_days[df['machine_status'] == 0] = 0
    return recommended_days

def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("Predictive Maintenance Model")
    st.session_state.setdefault('current_step', 0)
    st.session_state.setdefault('file_uploader_counter', 0)
    st.session_state.setdefault('df', None)
    st.session_state.setdefault('selected_features', [])
    st.session_state.setdefault('X', [])
    st.session_state.setdefault('y', [])
    st.session_state.setdefault('button_click', False)
    st.session_state.setdefault('button_click1', False)
    st.session_state.setdefault('selected_feature_selection_method', "Random Forest Importance")
    st.session_state.setdefault('feature_selection_threshold', 0.01)
    st.session_state.setdefault('selected_model', "Logistic Regression")
    st.session_state.setdefault('X_whole', [])

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
        machine_status_counts = st.session_state.df['machine_status'].value_counts()
        st.write(machine_status_counts)

        # Plotting the distribution of the target variable
        plt.figure(figsize=(8, 6))
        sns.countplot(data=st.session_state.df, x='machine_status')
        plt.title("Distribution of machine_status")
        plt.xlabel("Machine Status")
        plt.ylabel("Count")
        st.pyplot()

        # Numeric features summary statistics
        st.write("Summary statistics for numeric features:")
        st.write(st.session_state.df.describe())

        numeric_columns = st.session_state.df.select_dtypes(include=['number']).columns
        corr = get_corr(st.session_state.df[numeric_columns])
        # Correlation heatmap for numeric features
        st.write("Correlation Heatmap for numeric features:")
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        st.pyplot()

    
        # Step 2: Feature Selection
        st.subheader("Feature Selection")
        st.session_state.selected_feature_selection_method = st.selectbox("Select feature selection method", ["Logistic Regression Coefficients","Random Forest Importance", "SVM Weight Coefficients"])

        # Select feature selection threshold
        st.session_state.feature_selection_threshold = st.slider("Select feature selection threshold", min_value=0.0, max_value=1.0, value=0.01, step=0.01)

        if st.button("Next"):
            st.session_state.button_click = True

    if st.session_state.button_click:
        X, y, selected_features, X_whole = preprocess_data(st.session_state.df, st.session_state.selected_feature_selection_method, st.session_state.feature_selection_threshold)
        st.session_state.selected_features = selected_features
        st.session_state.X = X
        st.session_state.y = y
        st.session_state.X_whole = X_whole
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

        # Train the model and evaluate
        y_pred,y_test,accuracy,probabilities,total_pred = train_and_evaluate(X_train, y_train, X_test, y_test, st.session_state.selected_model,st.session_state.X_whole)

        # Display average accuracy
        st.write("Accuracy:", accuracy)

        # Classification Report
        st.write("Classification Report:")
        st.write(classification_report(y_test, y_pred))

        # Confusion Matrix
        st.write("Confusion Matrix:")
        conf_matrix = get_confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, cmap='coolwarm', fmt='d')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        st.pyplot()
        recommended_days = generate_recommended_days(probabilities,st.session_state.df)
        st.session_state.df['Days_to_service'] = recommended_days
        if st.button("Download CSV"):
            csv_file = st.session_state.df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_file,
                file_name="maintenance_recommendations.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
