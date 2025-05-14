import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import json
import hashlib
import os
from datetime import datetime
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, classification_report, 
                           confusion_matrix, roc_curve, auc, precision_recall_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

# -------------------- 🔐 Authentication System --------------------
USERS_FILE = "users.json"

def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password):
    users = load_users()
    if username in users:
        return False
    users[username] = hash_password(password)
    save_users(users)
    return True

def login_user(username, password):
    users = load_users()
    return username in users and users[username] == hash_password(password)

# -------------------- 🛠️ Utility Functions --------------------
def get_current_datetime():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join("uploads", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return False

# -------------------- 📊 Data Processing --------------------
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def preprocess_data(data, target_column):
    # Basic cleaning
    data = data.dropna()
    data = data.drop_duplicates()
    
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    return X, y

def balance_data(X, y, method='undersample'):
    if method == 'undersample':
        rus = RandomUnderSampler(random_state=42)
        X_res, y_res = rus.fit_resample(X, y)
    elif method == 'oversample':
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
    else:
        X_res, y_res = X, y
    
    return X_res, y_res

# -------------------- 🤖 Model Training --------------------
def train_models(X_train, X_test, y_train, y_test, feature_names):
    results = {}
    
    # Initialize models with optimized parameters
    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        
    }
    
    # Train and evaluate each model
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [0]*len(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # ROC Curve metrics
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'f1_score': f1,
                'classification_report': report,
                'fpr': fpr,
                'tpr': tpr,
                'roc_auc': roc_auc,
                'precision': precision,
                'recall': recall,
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'feature_names': feature_names
            }
        except Exception as e:
            st.error(f"Error training {name}: {str(e)}")
            continue
    
    return results

# -------------------- 📈 Visualization --------------------
def plot_metrics(results):
    # Set up figure
    plt.figure(figsize=(15, 10))
    
    # ROC Curve
    plt.subplot(2, 2, 1)
    for name, res in results.items():
        plt.plot(res['fpr'], res['tpr'], label=f"{name} (AUC = {res['roc_auc']:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    
    # Precision-Recall Curve
    plt.subplot(2, 2, 2)
    for name, res in results.items():
        plt.plot(res['recall'], res['precision'], label=name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    
    # Feature Importance (for tree-based models)
    plt.subplot(2, 2, 3)
    for name, res in results.items():
        if hasattr(res['model'], 'feature_importances_') and hasattr(res, 'feature_names'):
            try:
                importances = pd.Series(res['model'].feature_importances_, index=res['feature_names'])
                importances.nlargest(10).plot(kind='barh')
                plt.title(f'{name} - Feature Importance')
                break
            except Exception as e:
                st.warning(f"Could not plot feature importance: {str(e)}")
                break
    
    # Confusion Matrix for best model
    plt.subplot(2, 2, 4)
    if results:
        best_model = max(results.items(), key=lambda x: x[1]['f1_score'])[0]
        cm = results[best_model]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {best_model}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
    
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

# -------------------- 🚀 Main Application --------------------
def main():
    st.set_page_config(
        page_title="FraudGuard: Credit Card Fraud Detection System",
        page_icon="💳",
        layout="wide"
    )
    
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'feature_names' not in st.session_state:
        st.session_state.feature_names = None
    
    # Authentication
    if not st.session_state.logged_in:
        auth_page()
        return
    
    # Main application
    st.title("💳 FraudGuard: Credit Card Fraud Detection System")
    st.write(f"Welcome, {st.session_state.username}! | Session started at: {get_current_datetime()}")
    
    # Sidebar navigation
    menu = ["Data Upload", "Data Exploration", "Model Training", "Model Evaluation", "Predictions"]
    choice = st.sidebar.selectbox("Navigation", menu)
    
    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.data = None
        st.session_state.results = None
        st.rerun()
    
    # Page routing
    if choice == "Data Upload":
        data_upload_page()
    elif choice == "Data Exploration":
        data_exploration_page()
    elif choice == "Model Training":
        model_training_page()
    elif choice == "Model Evaluation":
        model_evaluation_page()
    elif choice == "Predictions":
        prediction_page()

def auth_page():
    st.title("🔐 FraudGuard Authentication")
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login to Your Account")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if login_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")
    
    with tab2:
        st.subheader("Create New Account")
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")
        confirm_pass = st.text_input("Confirm Password", type="password")
        
        if st.button("Register"):
            if new_pass != confirm_pass:
                st.error("Passwords do not match!")
            elif len(new_pass) < 8:
                st.error("Password must be at least 8 characters")
            elif create_user(new_user, new_pass):
                st.success("Account created successfully! Please login.")
            else:
                st.error("Username already exists")

def data_upload_page():
    st.header("📁 Data Upload")
    st.write("Upload your credit card transaction dataset for fraud detection analysis.")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file", 
        type=["csv", "xls", "xlsx"],
        help="Dataset should contain transaction details including a target column indicating fraud (1) or not fraud (0)"
    )
    
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            st.session_state.data = data
            st.success("Data loaded successfully!")
            
            # Show basic info
            st.subheader("Dataset Overview")
            st.write(f"Shape: {data.shape[0]} rows, {data.shape[1]} columns")
            st.write("First 5 rows:")
            st.dataframe(data.head())
            
            # Select target column
            target_col = st.selectbox(
                "Select the target column (fraud indicator)",
                data.columns,
                index=len(data.columns)-1 if 'Class' in data.columns else 0
            )
            st.session_state.target_col = target_col
            st.session_state.feature_names = data.drop(columns=[target_col]).columns.tolist()

def data_exploration_page():
    if st.session_state.data is None:
        st.warning("Please upload data first!")
        return
    
    st.header("🔍 Data Exploration")
    data = st.session_state.data
    
    # Basic statistics
    st.subheader("Basic Statistics")
    st.write(data.describe())
    
    # Target distribution
    st.subheader("Target Distribution")
    target_counts = data[st.session_state.target_col].value_counts()
    st.bar_chart(target_counts)
    st.write(f"Fraudulent transactions: {target_counts.get(1, 0)} ({target_counts.get(1, 0)/len(data)*100:.2f}%)")
    st.write(f"Legitimate transactions: {target_counts.get(0, 0)} ({target_counts.get(0, 0)/len(data)*100:.2f}%)")
    
    # Correlation matrix
    st.subheader("Correlation Matrix")
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr = data[numeric_cols].corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
        st.pyplot(plt)
        plt.close()
    else:
        st.warning("Not enough numeric columns for correlation analysis")

def model_training_page():
    if st.session_state.data is None:
        st.warning("Please upload data first!")
        return
    
    st.header("🤖 Model Training")
    data = st.session_state.data
    target_col = st.session_state.target_col
    
    # Train/test split
    test_size = st.slider("Test Set Size (%)", 10, 40, 20)
    
    if st.button("Train Models"):
        with st.spinner("Training models... This may take a few minutes wait ya handsa"):
            try:
                # Preprocess data
                X, y = preprocess_data(data, target_col)
                
                # Balance classes (using undersampling by default)
                X_res, y_res = balance_data(X, y, method='undersample')
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_res, y_res, 
                    test_size=test_size/100, 
                    random_state=42,
                    stratify=y_res
                )
                
                # Scale features (using StandardScaler by default)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train models
                results = train_models(X_train_scaled, X_test_scaled, y_train, y_test, st.session_state.feature_names)
                
                if not results:
                    st.error("No models were successfully trained!")
                    return
                
                st.session_state.results = results
                st.session_state.scaler = scaler
                
                st.success("Model training completed!")
                
                # Show summary results
                st.subheader("Model Performance Summary")
                summary_df = pd.DataFrame({
                    'Model': list(results.keys()),
                    'Accuracy': [res['accuracy'] for res in results.values()],
                    'F1 Score': [res['f1_score'] for res in results.values()],
                    'ROC AUC': [res['roc_auc'] for res in results.values()]
                })
                st.dataframe(summary_df.sort_values('F1 Score', ascending=False))
                
            except Exception as e:
                st.error(f"Error during model training: {str(e)}")

def model_evaluation_page():
    if st.session_state.results is None:
        st.warning("Please train models first!")
        return
    
    st.header("📊 Model Evaluation")
    results = st.session_state.results
    
    # Plot metrics
    st.subheader("Model Performance Metrics")
    plot_metrics(results)
    
    # Detailed reports
    st.subheader("Detailed Classification Reports")
    model_to_view = st.selectbox("Select model to view details", list(results.keys()))
    
    st.write(f"### {model_to_view} Performance")
    st.write(f"**Accuracy:** {results[model_to_view]['accuracy']:.4f}")
    st.write(f"**F1 Score:** {results[model_to_view]['f1_score']:.4f}")
    st.write(f"**ROC AUC:** {results[model_to_view]['roc_auc']:.4f}")
    
    st.write("**Classification Report:**")
    st.dataframe(pd.DataFrame(results[model_to_view]['classification_report']).transpose())
    
    # Confusion matrix
    st.write("**Confusion Matrix:**")
    cm = results[model_to_view]['confusion_matrix']
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(plt)
    plt.close()

def prediction_page():
    if st.session_state.results is None:
        st.warning("Please train models first!")
        return
    
    st.header("🔮 Make Predictions")
    results = st.session_state.results
    
    # Select model
    model_name = st.selectbox("Select model for predictions", list(results.keys()))
    model = results[model_name]['model']
    
    # Input form for prediction
    st.subheader("Enter Transaction Details")
    
    # Create input fields based on available features
    input_data = {}
    for feature in st.session_state.feature_names:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.write(f"**{feature}**")
        with col2:
            if st.session_state.data[feature].dtype in [np.float64, np.int64]:
                min_val = float(st.session_state.data[feature].min())
                max_val = float(st.session_state.data[feature].max())
                default_val = float(st.session_state.data[feature].median())
                input_data[feature] = st.number_input(
                    f"Enter {feature}", 
                    min_value=min_val, 
                    max_value=max_val, 
                    value=default_val,
                    label_visibility="collapsed"
                )
            else:
                input_data[feature] = st.text_input(
                    f"Enter {feature}", 
                    value="",
                    label_visibility="collapsed"
                )
    
    if st.button("Predict Fraud Risk"):
        try:
            # Prepare input data
            input_df = pd.DataFrame([input_data])
            
            # Apply scaling if used during training
            if hasattr(st.session_state, 'scaler') and st.session_state.scaler is not None:
                input_df = st.session_state.scaler.transform(input_df)
            
            # Make prediction
            prediction = model.predict(input_df)
            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(input_df)[0][1]
            else:
                probability = 1.0 if prediction[0] == 1 else 0.0
            
            # Display results
            if prediction[0] == 1:
                st.error(f"🚨 Fraud Detected! (Probability: {probability:.2%})")
            else:
                st.success(f"✅ Legitimate Transaction (Probability: {1-probability:.2%})")
            
            # Show probability gauge
            st.write("Fraud Probability:")
            st.progress(int(probability * 100))
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

# -------------------- � Run the App --------------------
if __name__ == "__main__":
    # Create uploads directory if it doesn't exist
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    
    main()