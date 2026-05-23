import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

RANDOM_SEED = 42
DATASET_PATH = "./data/heart_disease_uci.csv"

NUMERICAL_FEATURES = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
CATEGORICAL_FEATURES = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
TARGET_FEATURE = 'num'

def load_and_prepare_data(filepath):
    """Loads dataset and flattens the target variable to binary."""
    df = pd.read_csv(filepath)
    
    # Flatten the 1-4 num values to 1 to frame as a Binary Classification Problem
    # 0 remains 0 (no disease), while 1, 2, 3, and 4 become 1 (increased risk)
    df[TARGET_FEATURE] = df[TARGET_FEATURE].apply(lambda x: 1 if x > 0 else 0)

    # casting boolean to string as a 3rd type 'Null' is later added to fill the blanks 
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].apply(lambda x: str(x) if pd.notnull(x) else np.nan)
    
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_FEATURE]
    
    return X, y

def preprocessing_pipeline():
    
    # Numerical Preprocessing: Median Imputation followed by Z-score normalization.
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical Preprocessing: 'Null' filling followed by OneHotEncoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Null')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine transformers into a single preprocessor.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ])
        
    return preprocessor

def bootcamp():
    # data loading, preprocessing 
    X, y = load_and_prepare_data(DATASET_PATH)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    
    preprocessor = preprocessing_pipeline()
    
    # 2. model definition 
    models = {
        'LR (C=1.0, Neutral)': LogisticRegression(
            C=1.0, 
            class_weight=None, 
            random_state=RANDOM_SEED
        ),
        'LR (C=1.0, Balanced)': LogisticRegression(
            C=1.0, 
            class_weight='balanced', 
            random_state=RANDOM_SEED
        ),
        'LR (C=0.01, Neutral)': LogisticRegression(
            C=0.01, 
            class_weight=None, 
            random_state=RANDOM_SEED
        ),
        'LR (C=0.01, Balanced)': LogisticRegression(
            C=0.01, 
            class_weight='balanced', 
            random_state=RANDOM_SEED
        )
    }

    #output storing 
    data = []
    global_metrics = {}
    
    # 3. training + evaluation 
    for model_name, classifier in models.items():
        print(f"training {model_name}\n") 

        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])
        
        # Train the model
        model_pipeline.fit(X_train, y_train)
        
        # Predict: evaluation over the training set then testing set 
        y_pred_train = model_pipeline.predict(X_train)
        y_pred_test = model_pipeline.predict(X_test)

        # calculating the overfitting gap
        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test = accuracy_score(y_test, y_pred_test)

        overfit_gap = acc_train - acc_test

        report = classification_report(y_test, y_pred_test, output_dict=True)

        global_metrics[model_name] = {
            'Test Accuracy': acc_test,
            'Overfitting Gap': overfit_gap,
            'Precision (Risk)': report['1']['precision'],
            'Recall (Risk)': report['1']['recall'],
            'F1-Score (Risk)': report['1']['f1-score']
        }

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
        data.extend([
            {'Model': model_name, 'Metric': 'True Negatives', 'Count': tn},
            {'Model': model_name, 'Metric': 'False Positives', 'Count': fp},
            {'Model': model_name, 'Metric': 'False Negatives', 'Count': fn},
            {'Model': model_name, 'Metric': 'True Positives', 'Count': tp}
        ])


    df_cm = pd.DataFrame(data)
    
    global_df = pd.DataFrame(global_metrics)

    #printing model final chart : every configurations tested
    print(global_df.round(4).to_string())

    
    plt.figure(figsize=(10, 6))

    #grouped bar chart
    sns.barplot(data=df_cm, x='Metric', y='Count', hue='Model', palette='Set1')

    plt.title('Confusion Matrix Comparison: Patient Counts per Outcome')
    plt.ylabel('Number of Patients')
    plt.xlabel('Prediction Category')

    # Push the legend outside the plot so it doesn't block the bars
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Models')
    plt.tight_layout()

    plt.show()

if __name__=="__main__":
    bootcamp()
