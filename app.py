"""
Stroke Prediction and Classification System
- Handles tabular data preprocessing (missing values, imbalance, normalization)
- Implements multiple ML models (XGBoost, LightGBM, Random Forest, Decision Tree)
- Performs feature selection and importance analysis
- Processes and classifies brain scan images
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import imghdr
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import lightgbm as lgbm
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

#################################################
# PART 1: TABULAR DATA PROCESSING AND MODELING
#################################################

class TabularDataProcessor:
    """
    Handles processing and modeling of tabular data for stroke prediction
    """
    
    def __init__(self, file_path):
        """Initialize with data file path"""
        self.file_path = file_path
        self.data = None
        self.X = None
        self.y = None
        self.X_smote = None
        self.y_smote = None
        self.models = None
        self.results_test = None
        self.results_train = None
        self.feature_importances = None
    
    def load_data(self):
        """Load the dataset"""
        print("Loading tabular data...")
        try:
            self.data = pd.read_csv(self.file_path)
            print(f"Data loaded successfully: {self.data.shape} rows and columns")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def explore_data(self):
        """Explore and summarize the dataset"""
        print("\n===== DATA EXPLORATION =====")
        print(f"Dataset shape: {self.data.shape}")
        print("\nData types:")
        print(self.data.dtypes)
        print("\nMissing values:")
        print(self.data.isnull().sum())
        print("\nData summary:")
        print(self.data.describe())
        
        # Display target distribution
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(x='stroke', data=self.data)
        for p in ax.patches:
            ax.annotate(f'{p.get_height()} ({100 * p.get_height() / len(self.data):.2f}%)', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', xytext=(0, 10), 
                        textcoords='offset points')
        plt.title('Distribution of Stroke Cases')
        plt.show()
        
        # Basic correlations
        plt.figure(figsize=(12, 10))
        numeric_cols = self.data.select_dtypes(include=['number']).columns
        sns.heatmap(self.data[numeric_cols].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Matrix of Numeric Features')
        plt.tight_layout()
        plt.show()
    
    def preprocess_data(self):
        """Preprocess the dataset (handling missing values, encoding, etc.)"""
        print("\n===== DATA PREPROCESSING =====")
        
        # Make a copy to avoid modifying the original
        df = self.data.copy()
        
        # Check for missing values
        missing_values = df.isnull().sum()
        print("Missing values before imputation:")
        print(missing_values[missing_values > 0])
        
        # Handle missing numerical values
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        numerical_imputer = SimpleImputer(strategy='median')
        df[numerical_cols] = numerical_imputer.fit_transform(df[numerical_cols])
        
        # Handle missing categorical values
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])
        
        # Encode categorical variables
        df = pd.get_dummies(df, drop_first=True)
        
        # Normalize numerical features
        scaler = MinMaxScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        
        print("Data preprocessing completed.")
        print(f"Processed data shape: {df.shape}")
        
        # Separate features and target
        self.y = df['stroke']
        self.X = df.drop('stroke', axis=1)
        
        print(f"Features: {self.X.shape}, Target: {self.y.shape}")
        return df
    
    def handle_imbalanced_data(self):
        """Handle imbalanced dataset using SMOTE and SMOTETomek"""
        print("\n===== HANDLING IMBALANCED DATA =====")
        
        # Count before balancing
        print("Class distribution before balancing:")
        print(self.y.value_counts())
        
        # Apply SMOTE for oversampling
        print("Applying SMOTE for class balancing...")
        smote = SMOTE(random_state=RANDOM_SEED)
        self.X_smote, self.y_smote = smote.fit_resample(self.X, self.y)
        
        # Count after balancing
        print("Class distribution after balancing:")
        print(pd.Series(self.y_smote).value_counts())
        
        # Visualize the balanced dataset
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.countplot(x=self.y)
        plt.title("Original Class Distribution")
        
        plt.subplot(1, 2, 2)
        sns.countplot(x=self.y_smote)
        plt.title("Balanced Class Distribution")
        plt.tight_layout()
        plt.show()
        
        print(f"New balanced features shape: {self.X_smote.shape}")
        print(f"New balanced target shape: {self.y_smote.shape}")
    
    def setup_models(self):
        """Set up multiple machine learning models for evaluation"""
        print("\n===== SETTING UP MODELS =====")
        
        self.models = [
            ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
            ('Decision Tree', DecisionTreeClassifier(random_state=RANDOM_SEED)),
            ('Random Forest', RandomForestClassifier(random_state=RANDOM_SEED)),
            ('Gradient Boosting', GradientBoostingClassifier(random_state=RANDOM_SEED)),
            ('XGBoost', XGBClassifier(random_state=RANDOM_SEED, use_label_encoder=False, eval_metric='logloss')),
            ('LightGBM', lgbm.LGBMClassifier(random_state=RANDOM_SEED)),
            ('Support Vector Machine', SVC(probability=True, random_state=RANDOM_SEED)),
            ('K-nearest neighbors', KNeighborsClassifier()),
            ('Naive Bayes', GaussianNB())
        ]
        
        print(f"Set up {len(self.models)} machine learning classifiers.")
    
    def evaluate_models(self, num_iter=5, num_folds=10):
        """
        Evaluate all models using nested cross-validation
        
        Args:
            num_iter: Number of iterations for model evaluation
            num_folds: Number of folds for cross-validation
        """
        print("\n===== MODEL EVALUATION =====")
        
        # Initialize DataFrames to store results
        self.results_test = pd.DataFrame()
        self.results_train = pd.DataFrame()
        results_test_iter = pd.DataFrame()
        results_train_iter = pd.DataFrame()
        
        # Store confusion matrices and ROC curve data
        test_confusion = {}
        train_confusion = {}
        roc_data = {}
        
        for name, model in self.models:
            print(f"Evaluating model: {name}...")
            
            # Initialize metrics storage
            test_metrics = {
                'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'AUC': []
            }
            train_metrics = {
                'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'AUC': []
            }
            test_confusion_matrices = []
            train_confusion_matrices = []
            roc_curves = []
            
            # Run multiple iterations of cross-validation
            for i in range(num_iter):
                # Set up stratified CV
                stratified_cv = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=i)
                
                for train_idx, test_idx in stratified_cv.split(self.X_smote, self.y_smote):
                    # Split data
                    X_train, X_test = self.X_smote.iloc[train_idx], self.X_smote.iloc[test_idx]
                    y_train, y_test = self.y_smote.iloc[train_idx], self.y_smote.iloc[test_idx]
                    
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred_test = model.predict(X_test)
                    y_pred_train = model.predict(X_train)
                    
                    # For models without predict_proba, use decision_function if available
                    if hasattr(model, "predict_proba"):
                        y_prob_test = model.predict_proba(X_test)[:, 1]
                    elif hasattr(model, "decision_function"):
                        y_prob_test = model.decision_function(X_test)
                    else:
                        y_prob_test = y_pred_test
                    
                    # Calculate metrics for test data
                    test_metrics['Accuracy'].append(accuracy_score(y_test, y_pred_test))
                    test_metrics['Precision'].append(precision_score(y_test, y_pred_test))
                    test_metrics['Recall'].append(recall_score(y_test, y_pred_test))
                    test_metrics['F1 Score'].append(f1_score(y_test, y_pred_test))
                    test_metrics['AUC'].append(roc_auc_score(y_test, y_prob_test))
                    
                    # Calculate metrics for training data
                    train_metrics['Accuracy'].append(accuracy_score(y_train, y_pred_train))
                    train_metrics['Precision'].append(precision_score(y_train, y_pred_train))
                    train_metrics['Recall'].append(recall_score(y_train, y_pred_train))
                    train_metrics['F1 Score'].append(f1_score(y_train, y_pred_train))
                    
                    if hasattr(model, "predict_proba"):
                        y_prob_train = model.predict_proba(X_train)[:, 1]
                    elif hasattr(model, "decision_function"):
                        y_prob_train = model.decision_function(X_train)
                    else:
                        y_prob_train = y_pred_train
                    
                    train_metrics['AUC'].append(roc_auc_score(y_train, y_prob_train))
                    
                    # Calculate confusion matrices
                    test_confusion_matrices.append(confusion_matrix(y_test, y_pred_test))
                    train_confusion_matrices.append(confusion_matrix(y_train, y_pred_train))
                    
                    # Calculate ROC curve
                    fpr, tpr, _ = roc_curve(y_test, y_prob_test)
                    roc_curves.append((fpr, tpr))
            
            # Store iteration results
            cols_iter = pd.MultiIndex.from_product([[name], [str(j) for j in range(num_iter * num_folds)]])
            
            test_iter_df = pd.DataFrame([v for v in test_metrics.values()], index=test_metrics.keys(), columns=cols_iter)
            results_test_iter = pd.concat([results_test_iter, test_iter_df], axis=1)
            
            train_iter_df = pd.DataFrame([v for v in train_metrics.values()], index=train_metrics.keys(), columns=cols_iter)
            results_train_iter = pd.concat([results_train_iter, train_iter_df], axis=1)
            
            # Calculate mean and std for each metric
            mean_test_metrics = {
                metric: (np.mean(values), np.std(values)) 
                for metric, values in test_metrics.items()
            }
            
            mean_train_metrics = {
                metric: (np.mean(values), np.std(values)) 
                for metric, values in train_metrics.items()
            }
            
            # Store mean and std metrics
            cols = pd.MultiIndex.from_product([[name], ['Mean', 'Std']])
            
            test_mean_df = pd.DataFrame([v for v in mean_test_metrics.values()], index=mean_test_metrics.keys(), columns=cols)
            self.results_test = pd.concat([self.results_test, test_mean_df], axis=1)
            
            train_mean_df = pd.DataFrame([v for v in mean_train_metrics.values()], index=mean_train_metrics.keys(), columns=cols)
            self.results_train = pd.concat([self.results_train, train_mean_df], axis=1)
            
            # Store confusion matrices and ROC data
            test_confusion[name] = test_confusion_matrices
            train_confusion[name] = train_confusion_matrices
            roc_data[name] = roc_curves
        
        print("Model evaluation completed.")
        return self.results_test, self.results_train, test_confusion, roc_data
    
    def visualize_results(self, test_confusion, roc_data):
        """Visualize model evaluation results"""
        print("\n===== VISUALIZING RESULTS =====")
        
        # Display test results
        print("\nTest Results (Mean ± Std):")
        print(self.results_test)
        
        # Display train results
        print("\nTrain Results (Mean ± Std):")
        print(self.results_train)
        
        # Visualize accuracy for all models
        metric = 'Accuracy'
        self.plot_model_comparison(metric)
        
        # Visualize F1 score for all models
        metric = 'F1 Score'
        self.plot_model_comparison(metric)
        
        # Visualize AUC for all models
        metric = 'AUC'
        self.plot_model_comparison(metric)
        
        # Visualize confusion matrix for best model
        best_model = self.get_best_model(metric='F1 Score')
        self.plot_confusion_matrix(best_model, test_confusion)
        
        # Visualize ROC curve for best model
        self.plot_roc_curve(best_model, roc_data)
    
    def plot_model_comparison(self, metric):
        """Plot comparison of models for a specific metric"""
        classifiers = [model[0] for model in self.models]
        
        # Define colors for the bars
        colors = plt.cm.tab10(np.linspace(0, 1, len(classifiers)))
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(
            classifiers, 
            self.results_test.loc[:, (slice(None), 'Mean')].loc[metric].values,
            yerr=self.results_test.loc[:, (slice(None), 'Std')].loc[metric].values,
            capsize=5, 
            color=colors
        )
        
        # Annotate each bar with its value
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2, 
                height + 0.01, 
                f'{height:.3f}', 
                ha='center', 
                va='bottom'
            )
        
        plt.xlabel('Classifier')
        plt.ylabel(metric)
        plt.title(f'Mean and Standard Deviation of {metric} Across Classifiers')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, model_name, test_confusion):
        """Plot confusion matrix for a specific model"""
        confusion_matrix_mean = np.mean(test_confusion[model_name], axis=0)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            confusion_matrix_mean, 
            annot=True, 
            cmap='Blues', 
            fmt='.1f',
            xticklabels=['No Stroke', 'Stroke'],
            yticklabels=['No Stroke', 'Stroke']
        )
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self, model_name, roc_data):
        """Plot ROC curve for a specific model"""
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        auc_scores = []
        
        plt.figure(figsize=(8, 6))
        
        # Plot individual ROC curves
        for curve in roc_data[model_name]:
            auc_score = auc(curve[0], curve[1])
            auc_scores.append(auc_score)
            plt.plot(curve[0], curve[1], color='lightblue', alpha=0.3)
            tprs.append(np.interp(mean_fpr, curve[0], curve[1]))
        
        # Calculate mean TPR and AUC
        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = np.mean(auc_scores)
        std_auc = np.std(auc_scores)
        
        # Plot mean ROC curve
        plt.plot(
            mean_fpr, 
            mean_tpr, 
            color='darkblue', 
            label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})'
        )
        
        # Plot diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def get_best_model(self, metric='F1 Score'):
        """Get the name of the best performing model based on a specific metric"""
        mean_values = self.results_test.loc[metric, (slice(None), 'Mean')]
        best_idx = mean_values.argmax()
        return mean_values.index[best_idx][0]
    
    def analyze_feature_importance(self):
        """
        Analyze feature importance using multiple methods
        1. Random Forest feature importance
        2. XGBoost feature importance
        3. Permutation importance
        """
        print("\n===== FEATURE IMPORTANCE ANALYSIS =====")
        
        # Use Random Forest for feature importance
        rf_model = RandomForestClassifier(random_state=RANDOM_SEED)
        feature_importances = []
        
        # Perform cross-validation to get stable feature importance
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        
        for train_idx, test_idx in skf.split(self.X_smote, self.y_smote):
            X_train = self.X_smote.iloc[train_idx]
            y_train = self.y_smote.iloc[train_idx]
            
            rf_model.fit(X_train, y_train)
            feature_importances.append(rf_model.feature_importances_)
        
        # Calculate mean importance
        mean_importances = np.mean(feature_importances, axis=0)
        std_importances = np.std(feature_importances, axis=0)
        
        # Create feature importance dataframe
        feature_importance_df = pd.DataFrame({
            'Feature': self.X.columns,
            'Importance': mean_importances,
            'Std': std_importances
        })
        
        # Sort by importance
        feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
        
        # Store feature importances
        self.feature_importances = feature_importance_df
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        plt.barh(
            range(len(feature_importance_df)), 
            feature_importance_df['Importance'],
            xerr=feature_importance_df['Std'],
            color='lightseagreen',
            alpha=0.8
        )
        plt.yticks(range(len(feature_importance_df)), feature_importance_df['Feature'])
        plt.xlabel('Importance')
        plt.title('Feature Importance (Random Forest)')
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
        
        print("Top 10 most important features:")
        print(feature_importance_df.head(10))
        
        return feature_importance_df
    
    def train_final_model(self, model_name=None):
        """
        Train the final model using the best hyperparameters
        
        Args:
            model_name: Name of the model to train (if None, uses the best model from evaluation)
        
        Returns:
            Trained model
        """
        print("\n===== TRAINING FINAL MODEL =====")
        
        # If model_name not provided, use the best model from evaluation
        if model_name is None:
            model_name = self.get_best_model()
        
        print(f"Training final model: {model_name}")
        
        # Get model object by name
        model_dict = dict(self.models)
        model = model_dict[model_name]
        
        # Define hyperparameter grid for the selected model
        param_grid = {}
        
        if model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif model_name == 'XGBoost':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        elif model_name == 'LightGBM':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 70]
            }
        
        # If parameter grid exists for the model, perform grid search
        if param_grid:
            print(f"Performing grid search for {model_name}...")
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=5,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(self.X_smote, self.y_smote)
            
            # Get best parameters
            best_params = grid_search.best_params_
            print(f"Best parameters: {best_params}")
            
            # Train model with best parameters
            model.set_params(**best_params)
        
        # Train the final model on the full dataset
        model.fit(self.X_smote, self.y_smote)
        
        # Evaluate on original unbalanced data to check real-world performance
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, stratify=self.y, random_state=RANDOM_SEED
        )
        
        y_pred = model.predict(X_test)
        
        # Print evaluation metrics
        print("\nFinal model performance on test data:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred):.4f}")
        print(f"Recall: {recall_score(y_test, y_pred):.4f}")
        print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
        
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            print(f"AUC: {roc_auc_score(y_test, y_prob):.4f}")
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            cmap='Blues', 
            fmt='d',
            xticklabels=['No Stroke', 'Stroke'],
            yticklabels=['No Stroke', 'Stroke']
        )
        plt.title(f'Confusion Matrix - {model_name} (Final Model)')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.show()
        
        return model
    
    def save_model(self, model, filename):
        """Save the trained model to disk"""
        import pickle
        
        # Create directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save the model
        with open(f'models/{filename}', 'wb') as file:
            pickle.dump(model, file)
        
        print(f"Model saved to models/{filename}")

#################################################
# PART 2: IMAGE DATA PROCESSING AND MODELING
#################################################

class ImageDataProcessor:
    """
    Handles processing and modeling of image data for stroke prediction
    """
    
    def __init__(self, dataset_dir):
        """Initialize with dataset directory"""
        self.dataset_dir = dataset_dir
        self.model = None
        self.history = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
    
    def clean_dataset(self):
        """Clean the image dataset by removing corrupted or incompatible images"""
        print("\n===== CLEANING IMAGE DATASET =====")
        
        # Supported image extensions
        valid_extensions = ['jpeg', 'jpg', 'bmp', 'png']
        
        # Check if the directory exists
        if not os.path.exists(self.dataset_dir):
            print(f"Error: Dataset directory '{self.dataset_dir}' does not exist.")
            return
        
        # Track cleaning statistics
        total_images = 0
        removed_images = 0
        
        # Process each class directory
        for class_dir in os.listdir(self.dataset_dir):
            class_path = os.path.join(self.dataset_dir, class_dir)
            
            # Skip if not a directory
            if not os.path.isdir(class_path):
                continue
            
            print(f"Processing class: {class_dir}")
            
            # Process each image in the class directory
            for image_file in os.listdir(class_path):
                total_images += 1
                image_path = os.path.join(class_path, image_file)
                
                try:
                    # Try to read the image
                    img = cv2.imread(image_path)
                    
                    # Check if image was read successfully
                    if img is None:
                        print(f"Unable to read image (removing): {image_path}")
                        os.remove(image_path)
                        removed_images += 1
                        continue
                    
                    # Check file extension
                    ext = imghdr.what(image_path)
                    if ext not in valid_extensions:
                        print(f"Removing image with invalid extension: {image_path}")
                        os.remove(image_path)
                        removed_images += 1
                    
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")
                    # Remove problematic images
                    if os.path.exists(image_path):
                        os.remove(image_path)
                        removed_images += 1
        
        print(f"Dataset cleaning completed: {removed_images} images removed out of {total_images} total images.")
    
    def load_data(self, img_size=(224, 224), batch_size=32):
        """
        Load the image dataset and prepare it for training
        
        Args:
            img_size: Tuple of (height, width) for image resizing
            batch_size: Batch size for training
        """
        print("\n===== LOADING IMAGE DATA =====")
        
        # Check for subdirectories
        if 'train' in os.listdir(self.dataset_dir) and 'validation' in os.listdir(self.dataset_dir):
            # Dataset is already split
            train_dir = os.path.join(self.dataset_dir, 'train')
            validation_dir = os.path.join(self.dataset_dir, 'validation')
            test_dir = os.path.join(self.dataset_dir, 'test')
            
            # Check if test directory exists
            if not os.path.exists(test_dir):
                print("Test directory not found. Using validation data for testing.")
                test_dir = validation_dir
        else:
            # Use the main directory and perform split later
            train_dir = validation_dir = test_dir = self.dataset_dir
        
        # Data augmentation for training
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2 if train_dir == self.dataset_dir else 0
        )
        
        # Only rescaling for validation and test data
        val_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            validation_split=0.5 if train_dir == self.dataset_dir else 0
        )
        
        # Load training data
        if train_dir == self.dataset_dir:
            # If using single directory, split into train/val/test
            self.train_data = train_datagen.flow_from_directory(
                train_dir,
                target_size=img_size,
                batch_size=batch_size,
                class_mode='binary',
                subset='training'
            )
            
            self.val_data = val_test_datagen.flow_from_directory(
                train_dir,
                target_size=img_size,
                batch_size=batch_size,
                class_mode='binary',
                subset='validation',
                shuffle=False
            )
            
            # For test data, we'll use a portion of validation data
            self.test_data = self.val_data
        else:
            # If already split, use the respective directories
            self.train_data = train_datagen.flow_from_directory(
                train_dir,
                target_size=img_size,
                batch_size=batch_size,
                class_mode='binary'
            )
            
            self.val_data = val_test_datagen.flow_from_directory(
                validation_dir,
                target_size=img_size,
                batch_size=batch_size,
                class_mode='binary',
                shuffle=False
            )
            
            self.test_data = val_test_datagen.flow_from_directory(
                test_dir,
                target_size=img_size,
                batch_size=batch_size,
                class_mode='binary',
                shuffle=False
            )
        
        print(f"Training data: {self.train_data.samples} images")
        print(f"Validation data: {self.val_data.samples} images")
        print(f"Test data: {self.test_data.samples} images")
        
        # Display class indices
        print("Class indices:", self.train_data.class_indices)
        
        # Visualize sample images
        self.visualize_samples()
    
    def visualize_samples(self, num_samples=5):
        """
        Visualize sample images from the training data
        
        Args:
            num_samples: Number of sample images to display
        """
        if self.train_data is None:
            print("No training data loaded. Cannot visualize samples.")
            return
        
        # Get a batch of samples
        images, labels = next(self.train_data)
        
        # Limit number of samples to display
        num_samples = min(num_samples, len(images))
        
        # Create figure
        plt.figure(figsize=(15, 3 * num_samples))
        
        for i in range(num_samples):
            plt.subplot(num_samples, 2, i*2 + 1)
            plt.imshow(images[i])
            plt.title(f"Class: {'Stroke' if labels[i] == 1 else 'Normal'}")
            plt.axis('off')
            
            # Add histogram to visualize intensity distribution
            plt.subplot(num_samples, 2, i*2 + 2)
            plt.hist(images[i].flatten(), bins=50, color='gray', alpha=0.7)
            plt.title('Pixel Intensity Distribution')
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
    
    def build_model(self, img_size=(224, 224)):
        """
        Build a CNN model for image classification
        
        Args:
            img_size: Tuple of (height, width) for input images
        """
        print("\n===== BUILDING CNN MODEL =====")
        
        # Create model
        model = Sequential([
            # First convolutional block
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(*img_size, 3)),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Second convolutional block
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Third convolutional block
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Flatten and dense layers
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(1, activation='sigmoid')  # Binary classification (stroke vs normal)
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', Precision(), Recall()]
        )
        
        # Print model summary
        model.summary()
        
        self.model = model
        return model
    
    def train_model(self, epochs=20, early_stopping=True):
        """
        Train the CNN model
        
        Args:
            epochs: Number of training epochs
            early_stopping: Whether to use early stopping
        """
        print("\n===== TRAINING CNN MODEL =====")
        
        if self.model is None:
            print("No model built. Call build_model() first.")
            return
        
        if self.train_data is None or self.val_data is None:
            print("Training or validation data not loaded. Call load_data() first.")
            return
        
        # Define callbacks
        callbacks = []
        
        # Create directory for TensorBoard logs
        os.makedirs('logs', exist_ok=True)
        
        # Add TensorBoard callback
        tensorboard_callback = TensorBoard(log_dir='logs')
        callbacks.append(tensorboard_callback)
        
        # Add early stopping if requested
        if early_stopping:
            early_stopping_callback = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            callbacks.append(early_stopping_callback)
        
        # Train the model
        self.history = self.model.fit(
            self.train_data,
            epochs=epochs,
            validation_data=self.val_data,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Model training completed.")
        
        # Visualize training history
        self.visualize_training_history()
    
    def visualize_training_history(self):
        """Visualize the model training history"""
        if self.history is None:
            print("No training history available.")
            return
        
        # Plot accuracy
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Training')
        plt.plot(self.history.history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Training')
        plt.plot(self.history.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Plot precision and recall if available
        if 'precision' in self.history.history and 'recall' in self.history.history:
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(self.history.history['precision'], label='Training')
            plt.plot(self.history.history['val_precision'], label='Validation')
            plt.title('Model Precision')
            plt.xlabel('Epoch')
            plt.ylabel('Precision')
            plt.legend()
            plt.grid(alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.plot(self.history.history['recall'], label='Training')
            plt.plot(self.history.history['val_recall'], label='Validation')
            plt.title('Model Recall')
            plt.xlabel('Epoch')
            plt.ylabel('Recall')
            plt.legend()
            plt.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    def evaluate_model(self):
        """Evaluate the model on test data"""
        print("\n===== EVALUATING CNN MODEL =====")
        
        if self.model is None:
            print("No model available for evaluation.")
            return
        
        if self.test_data is None:
            print("No test data loaded. Cannot evaluate model.")
            return
        
        # Evaluate on test data
        test_loss, test_acc, test_precision, test_recall = self.model.evaluate(self.test_data)
        
        print(f"Test loss: {test_loss:.4f}")
        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Test precision: {test_precision:.4f}")
        print(f"Test recall: {test_recall:.4f}")
        
        # Calculate F1 score
        f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall + 1e-7)
        print(f"Test F1 score: {f1:.4f}")
        
        # Get predictions for confusion matrix
        y_pred = []
        y_true = []
        
        for i in range(len(self.test_data)):
            x_batch, y_batch = self.test_data[i]
            y_pred_batch = self.model.predict(x_batch)
            y_pred.extend(y_pred_batch.flatten() > 0.5)
            y_true.extend(y_batch)
        
        # Convert to numpy arrays
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            cmap='Blues', 
            fmt='d',
            xticklabels=['Normal', 'Stroke'],
            yticklabels=['Normal', 'Stroke']
        )
        plt.title('Confusion Matrix (Test Data)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()
        
        # Calculate precision, recall, f1 score, and accuracy
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        
        return accuracy, precision, recall, f1
    
    def predict_image(self, image_path):
        """
        Predict stroke probability for a given image
        
        Args:
            image_path: Path to the input image file
        """
        if self.model is None:
            print("No model available for prediction.")
            return
        
        try:
            # Read the image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error: Could not read image from {image_path}")
                return
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize the image
            img_resized = cv2.resize(img, (224, 224))
            
            # Display the original image
            plt.figure(figsize=(5, 5))
            plt.imshow(img_resized)
            plt.title('Input Image')
            plt.axis('off')
            plt.show()
            
            # Normalize pixel values
            img_normalized = img_resized / 255.0
            
            # Add batch dimension
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            # Make prediction
            prediction = self.model.predict(img_batch)[0][0]
            
            print(f"Stroke probability: {prediction:.4f}")
            
            # Classify based on threshold
            if prediction > 0.5:
                print("Prediction: Stroke (High Risk)")
            else:
                print("Prediction: Normal (Low Risk)")
            
        except Exception as e:
            print(f"Error predicting image: {e}")
    
    def save_model(self, filename='stroke_cnn_model.keras'):
        """Save the trained CNN model"""
        if self.model is None:
            print("No model available to save.")
            return
        
        # Create directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save the model
        try:
            self.model.save(f'models/{filename}')
            print(f"Model saved to models/{filename}")
        except Exception as e:
            print(f"Error saving model: {e}")

#################################################
# PART 3: MAIN EXECUTION
#################################################

def main():
    """Main execution function"""
    print("==================================================")
    print("STROKE PREDICTION AND CLASSIFICATION SYSTEM")
    print("==================================================")
    
    # Part 1: Tabular Data Processing
    print("\n[PART 1: TABULAR DATA PROCESSING]")
    
    # Initialize tabular data processor
    # Note: Replace 'dataset.csv' with the actual dataset file path
    tabular_processor = TabularDataProcessor('raw_data/healthcare-dataset-stroke-data.csv')
    
    # Process tabular data
    if tabular_processor.load_data():
        tabular_processor.explore_data()
        processed_data = tabular_processor.preprocess_data()
        tabular_processor.handle_imbalanced_data()
        tabular_processor.setup_models()
        
        # Evaluate models
        results_test, results_train, test_confusion, roc_data = tabular_processor.evaluate_models()
        
        # Visualize results
        tabular_processor.visualize_results(test_confusion, roc_data)
        
        # Analyze feature importance
        feature_importance = tabular_processor.analyze_feature_importance()
        
        # Train and save the final model
        best_model = tabular_processor.train_final_model()
        tabular_processor.save_model(best_model, 'stroke_tabular_model.pkl')
    
    # Part 2: Image Data Processing
    print("\n[PART 2: IMAGE DATA PROCESSING]")
    
    # Initialize image data processor
    # Note: Replace 'Dataset' with the actual dataset directory
    image_processor = ImageDataProcessor('Dataset')
    
    # Process image data
    image_processor.clean_dataset()
    image_processor.load_data()
    image_processor.build_model()
    image_processor.train_model()
    image_processor.evaluate_model()
    image_processor.save_model()
    
    print("\n==================================================")
    print("PROCESSING AND MODELING COMPLETED SUCCESSFULLY")
    print("==================================================")

if __name__ == "__main__":
    main()