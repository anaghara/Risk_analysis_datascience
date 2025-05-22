#!/usr/bin/env python
# coding: utf-8

# ========== INTRODUCTION==========

# Construction projects involve multiple uncertainties such as budget overruns, schedule delays, and inaccurate cost estimates, making risk prediction and management critical. An effective risk assessment model helps stakeholders make data-driven decisions, reducing financial losses and improving project efficiency.
# 
# This project leverages machine learning techniques, specifically Random Forest, to predict project risk levels based on key factors such as Project Duration, Cost to Date, Estimated Duration, and Budget Overruns. The Gini Importance metric is utilized to ensure that only the most significant features contribute to risk assessment, making the model robust and reliable.
# 
# Why Use Gini Feature Importance?
# One of the challenges in risk prediction is identifying and prioritizing the most relevant factors while eliminating less significant ones. Gini Importance helps:
# - Focus on high-impact variables that strongly influence project risk.
# - Ensure that critical risk indicators are not ignored, leading to a well-balanced model.
# - Improve prediction accuracy by removing irrelevant features, optimizing model performance.
# 
# To enhance accuracy, a weighted risk scoring system is implemented, ensuring that risk factors are quantified effectively, allowing for early identification of high-risk projects.

#  ========== 1. IMPORT NECESSARY LIBRARIES ==========

# The code imports libraries for data handling (pandas, numpy), splitting datasets (train_test_split), feature processing (LabelEncoder, StandardScaler), and model training (RandomForestClassifier). It includes hyperparameter tuning (GridSearchCV), evaluation metrics (accuracy_score, precision_score, etc.), and visualization tools (matplotlib, seaborn).

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV


# ========== 2. DATA LOADING ==========

# The function `load_data()` reads the dataset construction_projects_data.csv using `pandas.read_csv()`. It checks if the file exists, prints a success message if loaded, or an error if missing. The function returns the dataframe `df` for further processing, ensuring smooth data handling for analysis.

# In[2]:


def load_data():
    """Loads data from the dataset file: construction_projects_data.csv."""
    file_path = "construction_projects_data.csv"  # Explicit dataset name
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset '{file_path}' loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

# Run Data Loading
df = load_data()


# ========== 3. DATA PREPARATION & DATA CLEANING ==========

# The function `preprocess_data(df)` cleans and prepares the dataset. It first checks if data is loaded, then drops unnecessary columns. Categorical columns are encoded using `LabelEncoder()`. Date columns are converted to datetime format. Missing values are filled with the median of numerical columns. The cleaned dataframe and label encoders are returned.

# In[3]:


def preprocess_data(df):
    """Preprocesses and cleans the data."""

    if df is None:  # Check if data loading was successful
        return None, None

    df.drop(columns=['Unnamed: 0', 'IDProject_'], inplace=True, errors='ignore')

    categorical_cols = ['Area_', 'SubArea_', 'Project_Category', 'Contract_Type_']
    for col in list(categorical_cols):  # Iterate over a copy to safely remove items
        if col not in df.columns:
            categorical_cols.remove(col)

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    date_cols = ['Start_Date_', 'EstimatedFinalCompletion_', 'Actual_End_Date_']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    df.fillna(df.median(numeric_only=True), inplace=True)

    return df, label_encoders

# Run Data Cleaning
df, label_encoders = preprocess_data(df)


# ========== 5. FEATURE ENGINEERING (Combined with Data Cleaning for efficiency) ==========

# The function `feature_engineering(df)` creates new features from existing data. It calculates `Project_Duration` and `Estimated_Duration` by finding the difference between dates. It determines `Budget_Overrun` and `Project_Delay` using cost and time comparisons. A `Risk_Factor` is assigned based on budget and delay status. Unnecessary columns are dropped, and the updated dataframe is returned.

# In[4]:


def feature_engineering(df):
    """Extracts new features based on existing data."""

    if df is None:
        return None

    # Compute Actual Project Duration
    if all(col in df.columns for col in ['Start_Date_', 'Actual_End_Date_']):
        df['Project_Duration'] = (df['Actual_End_Date_'] - df['Start_Date_']).dt.days

    # Compute Estimated Project Duration
    if all(col in df.columns for col in ['Start_Date_', 'EstimatedFinalCompletion_']):
        df['Estimated_Duration'] = (df['EstimatedFinalCompletion_'] - df['Start_Date_']).dt.days

    # Budget Overrun Calculation
    if all(col in df.columns for col in ['CostToDate_', 'OriginalEstimate_']):
        df['Budget_Overrun'] = (df['CostToDate_'] > df['OriginalEstimate_']).astype(int)

    # Project Delay Calculation
    if all(col in df.columns for col in ['Actual_End_Date_', 'EstimatedFinalCompletion_']):
        df['Project_Delay'] = (df['Actual_End_Date_'] > df['EstimatedFinalCompletion_']).astype(int)

    # Risk Factor Calculation - Based on Multiple Factors
    if all(col in df.columns for col in ['Project_Duration', 'Estimated_Duration', 'Budget_Overrun', 'Project_Delay']):
        df['Risk_Factor'] = (
            ((df['Project_Duration'] > df['Estimated_Duration']) & (df['Project_Delay'] == 1)) |  # Project delayed beyond estimate
            (df['Budget_Overrun'] == 1) |  # Project exceeded budget
            ((df['Project_Duration'] > df['Estimated_Duration'] * 1.2))  # Project duration exceeded estimated by 20%
        ).astype(int)

    # Drop Unnecessary Columns
    df.drop(columns=['Start_Date_', 'EstimatedFinalCompletion_', 'Actual_End_Date_', 'projectPhase_', 
                     'Budget_Overrun', 'Project_Delay'], inplace=True, errors='ignore')

    return df

# Run Feature Engineering
df = feature_engineering(df)


# ========== # 6. DATASET SPLITTING ==========

# The function `split_data(df)` separates the dataset into features (`X`) and target (`y`), where `Risk_Factor` is the target variable. It uses `train_test_split()` to divide the data into training (80%) and testing (20%) sets with a fixed random state for reproducibility. The function returns `X_train`, `X_test`, `y_train`, and `y_test`.

# In[5]:


def split_data(df):
    """Splits dataset into features (X) and target (y)."""
    X = df.drop(columns=['Risk_Factor'], errors='ignore')
    y = df['Risk_Factor']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Run Data Splitting
X_train, X_test, y_train, y_test = split_data(df)


# ========== # 7. FEATURE SCALING ==========

# The function `scale_data(X_train, X_test)` standardizes numerical features using `StandardScaler`. It first converts input data to DataFrames, fills missing values with the median, and then applies `StandardScaler` to normalize the data. The function returns the scaled `X_train`, `X_test`, and the scaler instance for consistency in future transformations.

# In[6]:


def scale_data(X_train, X_test):
    """Scales numerical features using StandardScaler and handles NaN values."""
    
    # Convert to DataFrame (important for column operations)
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    
    # Handle missing values before scaling
    X_train.fillna(X_train.median(), inplace=True)
    X_test.fillna(X_test.median(), inplace=True)
    
    # Apply StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, scaler  # Return scaler for consistency

# Run Feature Scaling (After Fix)
X_train, X_test, scaler = scale_data(X_train, X_test)


# ========== # 8. BUILDING OR TRAINING MODEL ==========

# The function `train_model(X_train, y_train)` trains a `RandomForestClassifier`. It first converts `X_train` to a DataFrame and fills any missing values with the median. Then, it initializes a `RandomForestClassifier` with a fixed random state and fits it to the training data. The trained model is returned for making predictions.

# In[7]:


def train_model(X_train, y_train):
    """Trains a Random Forest Classifier model after handling missing values."""
    
    # Handle missing values in X_train
    X_train = pd.DataFrame(X_train)  # Convert back to DataFrame for column operations
    X_train.fillna(X_train.median(), inplace=True)  # Fill NaN with median values

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

# Run Model Training (After Fix)
model = train_model(X_train, y_train)


# ========== # 10. PREDICTION ==========

# The function `predict(model, X_test)` makes predictions using the trained model. It converts `X_test` to a DataFrame for consistency and then applies the model's `predict()` method to generate predictions. The function returns `y_pred`, which contains the predicted risk factors for the test dataset.

# In[8]:


def predict(model, X_test):
    """Makes predictions using the trained model after handling missing values."""

    # Convert X_test to DataFrame
    X_test = pd.DataFrame(X_test)

    # Handle missing values before prediction
    X_test.fillna(X_test.median(), inplace=True)

    # Make predictions
    y_pred = model.predict(X_test)
    return y_pred

# ... (Your model training code here) ...

# Assuming you have X_test and y_test (true labels) available
y_pred = predict(model, X_test)




# ========== # 11. HYPERPARAMETER TUNING ==========

# The function train_model(X_train, y_train) trains a RandomForestClassifier with hyperparameter tuning using GridSearchCV. It defines a parameter grid for tuning, initializes a base model, and performs a cross-validated grid search to find the best parameters. The function returns the optimized model with the highest accuracy.

# In[9]:


def train_model(X_train, y_train):
    """Trains a Random Forest Classifier model with hyperparameter tuning using GridSearchCV."""
    
    # Handle missing values in X_train
    X_train = pd.DataFrame(X_train)  # Convert back to DataFrame for column operations
    X_train.fillna(X_train.median(), inplace=True)  # Fill NaN with median values

    # Define parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Initialize base model
    rf_model = RandomForestClassifier(random_state=42)
    
    # Apply GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, 
                               cv=5, n_jobs=-1, scoring='accuracy', verbose=2)
    
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Accuracy Score: {grid_search.best_score_:.4f}")
    
    return best_model

# Run Model Training with Hyperparameter Tuning
model = train_model(X_train, y_train)


# ========== # 12. EVALUATION ==========

# The function `evaluate_model(y_test, y_pred)` assesses model performance using accuracy, precision, recall, and F1-score. It calculates these metrics using `sklearn.metrics`, stores them in a dictionary, and prints the results.
# 
# The evaluation results are:
# - Accuracy (86.32%): The model correctly predicts project risk in 86.32% of cases.
# - Precision (85.98%): Among all projects classified as high-risk, 86.32% were correctly identified.
# - Recall (86.32%): The model captures 86.32% of actual high-risk projects.
# - F1 Score (86.06%): A balanced measure combining precision and recall, indicating strong model performance.
# 
# These results suggest that the model effectively identifies potential project risks, with high accuracy and a good balance between precision and recall.

# In[10]:


def evaluate_model(y_test, y_pred):
    """Evaluates model performance."""
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    performance_metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }
    print("\nProject Risk Prediction Model Performance:")
    for metric, value in performance_metrics.items():
        print(f"{metric}: {value:.4f}")
    return performance_metrics

# Run Model Evaluation
performance_metrics = evaluate_model(y_test, y_pred)


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🚀 Step 3: Train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 🚀 Step 4: Make Predictions
y_pred = model.predict(X_test)  # Class predictions
y_prob = model.predict_proba(X_test)  # Probability predictions (Needed for ROC Curve)

# 🚀 Step 5: Model Evaluation Function
def evaluate_model(y_test, y_pred, y_prob):
    """Evaluates model performance and plots the ROC Curve."""
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    performance_metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }
    
    # Print Performance Metrics
    print("\nProject Risk Prediction Model Performance:")
    for metric, value in performance_metrics.items():
        print(f"{metric}: {value:.4f}")

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    
    if len(set(y_test)) == 2:  # Binary Classification
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])  # Probabilities for the positive class
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    else:  # Multi-class ROC (One-vs-Rest approach)
        for i in range(y_prob.shape[1]):
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, i], pos_label=i)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')

    # Formatting the ROC plot
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal reference line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()
    
    return performance_metrics

# 🚀 Step 6: Run Model Evaluation
performance_metrics = evaluate_model(y_test, y_pred, y_prob)


# ========== # 13. VISUALIZATION ==========

# The function computes feature importance using RandomForestClassifier. It prepares data by selecting numerical features and handling missing values. The model is trained using X_gini and y_gini, after which feature importance is extracted and sorted. This process helps identify key factors affecting project risks, enabling better decision-making and improving project management strategies.
# 
# Gini Importance Results:
# 
# - Project Duration (29.90%) – Longer projects tend to have higher risks.
# - Cost To Date (27.21%) – Projects with high costs are more likely to face risks.
# - Original Estimate (14.43%) – Budget estimation plays a crucial role in risk assessment.
# - Estimated Duration (11.70%) – Discrepancies between estimated and actual durations indicate potential risks.

# In[ ]:


# Compute Gini Feature Importance
df_gini = df.copy()
X_gini = df_gini.drop(columns=['Risk_Factor'], errors='ignore')
y_gini = df_gini['Risk_Factor']
X_gini = X_gini.select_dtypes(include=[np.number])
X_gini.fillna(X_gini.median(), inplace=True)

model_gini = RandomForestClassifier(random_state=42)
model_gini.fit(X_gini, y_gini)

feature_importance_gini = model_gini.feature_importances_
feature_names_gini = X_gini.columns
feature_importance_df_gini = pd.DataFrame({'Feature': feature_names_gini, 'Importance': feature_importance_gini})
feature_importance_df_gini = feature_importance_df_gini.sort_values(by='Importance', ascending=False)

print("\nGini Importance After Feature Extraction:")
print(feature_importance_df_gini)


# ========== # Main execution block ==========

# In[ ]:


def plot_feature_importance(feature_importance_df, title="Feature Importance"):
    """Plots feature importance."""
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Run Feature Importance Visualization
plot_feature_importance(feature_importance_df_gini, title="Gini Importance Before Training")


# ========== # 14.VISUAL INSIGHTS ON PROJECT RISK FACTORS ==========

# The following plots provide insights into project risks based on cost overruns, delays, and correlations between key factors.
# 
#  Risk Factor Distribution:
#     
# - Displays the number of high-risk (1) and low-risk (0) projects.
# - Helps understand the proportion of risky vs. non-risky projects.
# 
#  Budget Overrun vs. Risk Factor:
# 
# - A boxplot comparing project costs for low-risk and high-risk projects.
# - High-risk projects tend to have higher costs, showing budget overruns contribute significantly to risk.
# 
#  Project Delay vs. Risk Factor
# - Compares project duration for low-risk and high-risk projects.
# - Longer projects are more likely to be high-risk, reinforcing the importance of better scheduling.
# 
#  Correlation Heatmap
# - Displays relationships between different project factors.
# - Cost to Date and Original Estimate have strong correlations, indicating cost planning impacts risk.
# - Project Duration and Estimated Duration correlation suggests frequent project delays.
# 
#  Project Duration vs. Risk Factor
# - A histogram comparing project durations for high-risk and low-risk projects.
# - High-risk projects (red) tend to have longer durations.
# - Low-risk projects (blue) are more likely to be completed faster.

# In[ ]:


# 1. Risk Factor Distribution 
plt.figure(figsize=(8,5))
sns.countplot(x=df["Risk_Factor"], palette="coolwarm")
plt.title("Distribution of Risk Factors")
plt.xlabel("Risk Factor (0 = Low Risk, 1 = High Risk)")
plt.ylabel("Count of Projects")
plt.show()

# 2.Budget Overrun vs. Risk Factor 
plt.figure(figsize=(8,5))
sns.boxplot(x=df["Risk_Factor"], y=df["CostToDate_"], palette="coolwarm")
plt.title("Budget Overrun Impact on Risk")
plt.xlabel("Risk Factor (0 = Low Risk, 1 = High Risk)")
plt.ylabel("Cost to Date")
plt.show()

# 3. Project Delay vs. Risk Factor 
plt.figure(figsize=(8,5))
sns.boxplot(x=df["Risk_Factor"], y=df["Project_Duration"], palette="coolwarm")
plt.title("Project Delay Impact on Risk")
plt.xlabel("Risk Factor (0 = Low Risk, 1 = High Risk)")
plt.ylabel("Project Duration (Days)")
plt.show()

# 4. Correlation Heatmap 
plt.figure(figsize=(12,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap of Features")
plt.show()

# 5. Project Duration vs. Risk Factor 
plt.figure(figsize=(8,5))
sns.histplot(df[df["Risk_Factor"]==1]["Project_Duration"], bins=30, color="red", label="High Risk", kde=True)
sns.histplot(df[df["Risk_Factor"]==0]["Project_Duration"], bins=30, color="blue", label="Low Risk", kde=True)
plt.legend()
plt.title("Project Duration Distribution for High and Low-Risk Projects")
plt.xlabel("Project Duration (Days)")
plt.ylabel("Number of Projects")
plt.show()


# ========== # KEY INSIGHTS FROM PROJECT RISK ANALYSIS========== 

# 
# 1 . Budget Overrun ,Project duration and Project Delays are the primary drivers of project risk.
# 
# 2 . High-risk projects exceed cost estimates and face longer durations.
# 
# 3 . Strong correlation between Cost-to-Date and Original Estimate highlights budget planning concerns.
# 
# 4 . Delays cause operational inefficiencies, impacting multiple project phases.
# 
# 5 . Early risk detection is essential to minimize financial and scheduling failures.
