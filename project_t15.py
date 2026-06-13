import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, make_scorer, fbeta_score 
import seaborn as sns
import matplotlib.pyplot as plt

RANDOM_SEED = 42
DATASET_PATH = "./data/heart_disease_uci.csv"

NUMERICAL_FEATURES = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
CATEGORICAL_FEATURES = [['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal'],['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal', 'dataset']] #ablation study : adding / removing the 'dataset' columns.
TARGET_FEATURE = 'num'

def load_and_prepare_data(filepath, ablation):
    """Loads dataset and flattens the target variable to binary."""
    df = pd.read_csv(filepath)
    
    # Flatten the 1-4 num values to 1 to frame as a Binary Classification Problem
    # 0 remains 0 (no disease), while 1, 2, 3, and 4 become 1 (increased risk)
    df[TARGET_FEATURE] = df[TARGET_FEATURE].apply(lambda x: 1 if x > 0 else 0)

    # casting boolean to string as a 3rd type 'Null' is later added to fill the blanks 
    for col in CATEGORICAL_FEATURES[ablation]:
        df[col] = df[col].apply(lambda x: str(x) if pd.notnull(x) else np.nan)
    
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES[ablation]]
    y = df[TARGET_FEATURE]
    
    return X, y

def preprocessing_pipeline(ablation):
    
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

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES[ablation])
        ])
        
    return preprocessor
    
    

def evaluate_model_tournament(X_train, y_train, preprocessor):

    models_config = {
        'Logistic Regression': {
            'estimator': LogisticRegression(random_state=RANDOM_SEED, max_iter=1000),
            'params': {
                'classifier__C': [0.01, 0.1, 1.0, 10.0],
                'classifier__class_weight': [None, 'balanced']
            }
        },
        'Support Vector Machine': {
            'estimator': SVC(random_state=RANDOM_SEED),
            'params': {
                'classifier__C': [0.5, 1.0, 5.0], 
                'classifier__kernel': ['linear', 'rbf'],
                # Slightly tighter gamma constraints to balance the RBF boundary
                'classifier__gamma': ['scale', 0.05, 0.1],
                'classifier__class_weight': [None, 'balanced'] 
            }
        },
        'Random Forest': {
            'estimator': RandomForestClassifier(random_state=RANDOM_SEED),
            'params': {
                'classifier__n_estimators': [100, 200, 500],
                'classifier__max_depth': [4, 8, 10],
                'classifier__min_samples_split': [2, 5, 10],
                
                'classifier__min_samples_leaf': [6, 10],
                'classifier__class_weight': [None, 'balanced_subsample']
            }
        }
    }

    best_models = {}

    f2_scorer = make_scorer(fbeta_score, beta=2)

    print("starting hyperparameter sets tournament\n")
    
    for model_name, config in models_config.items():
        print(f"Training and Tuning: {model_name}...")
        
        # Create pipeline with the base estimator
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', config['estimator'])
        ])
        
        # GridSearchCV handles the 4-FCV on every hyperparameter combination
        grid_search = GridSearchCV(
            pipeline, 
            param_grid=config['params'], 
            cv=4, 
            scoring=f2_scorer, 
            n_jobs=-1, 
            verbose=1 
        )
        
        # Fit everything on the training set
        grid_search.fit(X_train, y_train)
        
        print(f" -> Best Hyperparameters: {grid_search.best_params_}")
        print(f" -> Best 4-FCV Validation F2 Score: {grid_search.best_score_:.4f}\n")
        
        # Store the overall best model
        best_models[model_name] = grid_search.best_estimator_

    return best_models

def bootcamp(ablation):
    # data loading, preprocessing 
    X, y = load_and_prepare_data(DATASET_PATH, ablation)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    
    preprocessor = preprocessing_pipeline(ablation)

    best_models = evaluate_model_tournament(X_train, y_train , preprocessor)
    
    
    #output storing 
    summary_data = []
    print("Final Evaluation on Unseen Test Set \n")
    
    summary_data = []
    cm_data = []
    
    for name, best_model in best_models.items():
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)

        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test = accuracy_score(y_test, y_pred_test)
        # Calculate F2 Score
        f2_train = fbeta_score(y_train, y_pred_train, beta=2)
        f2_test = fbeta_score(y_test, y_pred_test, beta=2)
        
        summary_data.append({
            'Model': name,
            'Train F2': round(f2_train, 4),
            'Test F2': round(f2_test, 4),
            'Train Acc': round(acc_train, 4),
            'Test Acc': round(acc_test, 4),
            'Overfit Gap': round(acc_train - acc_test, 4)
        })
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
        cm_data.extend([
            {'Model': name, 'Metric': 'True Negatives', 'Count': tn},
            {'Model': name, 'Metric': 'False Positives', 'Count': fp},
            {'Model': name, 'Metric': 'False Negatives', 'Count': fn},
            {'Model': name, 'Metric': 'True Positives', 'Count': tp}
        ])
        
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    df_cm = pd.DataFrame(cm_data)
    
    plt.figure(figsize=(10, 6))
    
    
    sns.barplot(data=df_cm, x='Metric', y='Count', hue='Model', palette='Set1')
    
    plt.title('Confusion Matrix Comparison: Patient Counts per Outcome')
    plt.ylabel('Number of Patients')
    plt.xlabel('Prediction Category')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Models')
    plt.tight_layout()
    
    plt.show()

if __name__=="__main__":
    bootcamp(1) #0 : no origin feature / 1 : origin feature included
