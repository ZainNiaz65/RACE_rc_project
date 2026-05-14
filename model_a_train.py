import pandas as pd
import numpy as np
import os
import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.semi_supervised import LabelPropagation
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                            recall_score, confusion_matrix, classification_report)
from sklearn.model_selection import cross_val_score, GridSearchCV
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import RACEPreprocessor


class ModelATrainer:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.training_times = {}
        
    def train_logistic_regression(self, X_train, y_train, class_weight=None):
        """
        Train Logistic Regression with hyperparameter tuning
        """
        print("\n" + "-"*60)
        print("Training Logistic Regression")
        print("-"*60)
        start_time = time.time()
        
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l2'],
            'solver': ['liblinear', 'lbfgs'],
            'max_iter': [1000]
        }
        
        lr = LogisticRegression(class_weight=class_weight, random_state=42)
        grid_search = GridSearchCV(lr, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        elapsed = time.time() - start_time
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        print(f"-- Completed in {elapsed/60:.2f} minutes")
        
        self.models['logistic_regression'] = best_model
        self.training_times['logistic_regression'] = elapsed
        return best_model
    
    def train_svm(self, X_train, y_train, class_weight=None):
        """
        Train SVM with hyperparameter tuning
        """
        print("\n" + "-"*60)
        print("Training SVM")
        print("-"*60)
        start_time = time.time()
        
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['linear'],
            'max_iter': [1000]
        }
        
        svm = SVC(class_weight=class_weight, random_state=42, probability=True)
        grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        elapsed = time.time() - start_time
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        print(f"-- Completed in {elapsed/60:.2f} minutes")
        
        self.models['svm'] = best_model
        self.training_times['svm'] = elapsed
        return best_model
    
    def train_random_forest(self, X_train, y_train, class_weight=None):
        """
        Train Random Forest
        """
        print("\n" + "-"*60)
        print("Training Random Forest")
        print("-"*60)
        start_time = time.time()
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf = RandomForestClassifier(class_weight=class_weight, random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        elapsed = time.time() - start_time
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        print(f"-- Completed in {elapsed/60:.2f} minutes")
        
        self.models['random_forest'] = best_model
        self.training_times['random_forest'] = elapsed
        return best_model
    
    def train_xgboost(self, X_train, y_train):
        """
        Train XGBoost (handles imbalance with scale_pos_weight)
        """
        print("\n" + "-"*60)
        print("Training XGBoost")
        print("-"*60)
        start_time = time.time()
        
        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
        scale_pos_weight = neg_count / max(pos_count, 1)
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0]
        }
        
        xgb_model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        elapsed = time.time() - start_time
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        print(f"-- Completed in {elapsed/60:.2f} minutes")
        
        self.models['xgboost'] = best_model
        self.training_times['xgboost'] = elapsed
        return best_model
    
    def train_naive_bayes(self, X_train, y_train):
        """
        Train Naive Bayes - works directly with sparse matrices
        """
        print("\n" + "-"*60)
        print("Training Naive Bayes")
        print("-"*60)
        start_time = time.time()
        
        # Naive Bayes works directly with sparse matrices
        nb = MultinomialNB(alpha=1.0)
        
        # Fit directly on sparse matrix
        nb.fit(X_train, y_train)
        elapsed = time.time() - start_time
        
        self.models['naive_bayes'] = nb
        self.training_times['naive_bayes'] = elapsed
        print(f"-- Completed in {elapsed/60:.2f} minutes")
        return nb
    
    def train_ensemble(self, X_train, y_train):
        """
        Train Voting Ensemble
        """
        print("\n" + "-"*60)
        print("Training Ensemble (Soft Voting)")
        print("-"*60)
        start_time = time.time()
        
        voting_models = []
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba') and name != 'ensemble_voting':
                voting_models.append((name, model))
        
        if len(voting_models) < 2:
            print("Need at least 2 models with predict_proba")
            return None
        
        ensemble = VotingClassifier(
            estimators=voting_models,
            voting='soft',
            weights=[1] * len(voting_models)
        )
        
        ensemble.fit(X_train, y_train)
        elapsed = time.time() - start_time
        
        self.models['ensemble_voting'] = ensemble
        self.training_times['ensemble_voting'] = elapsed
        print(f"-- Completed in {elapsed/60:.2f} minutes")
        return ensemble
    
    def unsupervised_kmeans_clustering(self, X_train, y_train):
        """
        K-Means clustering for exploring answer patterns
        """
        print("\n" + "-"*60)
        print("K-Means Clustering (Unsupervised)")
        print("-"*60)
        
        # Use subsample for clustering
        n_samples = min(50000, X_train.shape[0])
        indices = np.random.choice(X_train.shape[0], n_samples, replace=False)
        
        # Convert to dense for clustering if sparse
        if hasattr(X_train, 'toarray'):
            X_sample = X_train[indices].toarray()
        else:
            X_sample = X_train[indices]
        
        y_sample = y_train[indices] if len(y_train) >= n_samples else y_train
        
        # Find optimal k using elbow method
        inertias = []
        k_range = range(2, 11)
        
        print("Finding optimal number of clusters...")
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_sample)
            inertias.append(kmeans.inertia_)
            print(f"  k={k}, inertia={kmeans.inertia_:.0f}")
        
        # Fit with best k (using k=8 for demonstration)
        best_k = 8
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_sample)
        
        from sklearn.metrics import silhouette_score
        silhouette = silhouette_score(X_sample, clusters)
        print(f"\nSilhouette Score: {silhouette:.4f}")
        
        print("\nCluster Composition:")
        for cluster_id in range(best_k):
            cluster_mask = clusters == cluster_id
            cluster_samples = y_sample[cluster_mask]
            if len(cluster_samples) > 0:
                pos_pct = np.mean(cluster_samples) * 100
                print(f"  Cluster {cluster_id}: {len(cluster_samples)} samples, {pos_pct:.1f}% positive")
        
        self.models['kmeans'] = kmeans
        return kmeans, silhouette
    
    def semi_supervised_label_propagation(self, X_train, y_train, labeled_ratio=0.1):
        """
        Label Propagation with limited labeled data
        """
        print("\n" + "-"*60)
        print("Label Propagation (Semi-Supervised)")
        print("-"*60)
        print(f"Using only {labeled_ratio*100:.0f}% of labels")
        
        n_samples = len(y_train)
        n_labeled = int(n_samples * labeled_ratio)
        
        y_masked = np.full_like(y_train, -1, dtype=int)
        
        np.random.seed(42)
        labeled_indices = np.random.choice(n_samples, n_labeled, replace=False)
        y_masked[labeled_indices] = y_train[labeled_indices]
        
        # Convert to dense for Label Propagation if sparse
        if hasattr(X_train, 'toarray'):
            X_dense = X_train.toarray()
        else:
            X_dense = X_train
        
        lp = LabelPropagation(kernel='knn', n_neighbors=7, alpha=0.2)
        lp.fit(X_dense, y_masked)
        
        unlabeled_mask = y_masked == -1
        if np.any(unlabeled_mask):
            prop_accuracy = accuracy_score(
                y_train[unlabeled_mask], 
                lp.transduction_[unlabeled_mask]
            )
            print(f"Propagation Accuracy on unlabeled: {prop_accuracy:.4f}")
        
        self.models['label_propagation'] = lp
        return lp
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """
        Comprehensive evaluation
        """
        print(f"\n--- {model_name} Evaluation ---")
        
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives:  {cm[0,0]:6d}   False Positives: {cm[0,1]:6d}")
        print(f"  False Negatives: {cm[1,0]:6d}   True Positives:  {cm[1,1]:6d}")
        
        # Calculate specificity
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f"Specificity: {specificity:.4f}")
        
        # Save confusion matrix plot
        try:
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'Confusion Matrix - {model_name}')
            plt.tight_layout()
            plt.savefig(f'confusion_matrix_{model_name.replace(" ", "_")}.png')
            plt.close()
            print(f"Plot saved: confusion_matrix_{model_name.replace(' ', '_')}.png")
        except Exception as e:
            print(f"  Could not save plot: {e}")
        
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results = {
            'model': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'specificity': specificity,
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        self.results[model_name] = results
        return results
    
    def compare_all_models(self):
        """
        Print comparison of all trained models
        """
        print("\n" + "-"*80)
        print("Model Comparison Summary")
        print("-"*80)
        print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Time':<10}")
        print("-"*85)
        
        for name, results in self.results.items():
            train_time = self.training_times.get(name, 0)
            time_str = f"{train_time/60:.1f}min" if train_time > 0 else "N/A"
            print(f"{name:<25} {results['accuracy']:.4f}       "
                  f"{results['precision']:.4f}       "
                  f"{results['recall']:.4f}       "
                  f"{results['f1']:.4f}       {time_str:<10}")
        
        best_model = max(self.results.items(), key=lambda x: x[1]['f1'])
        print("-"*85)
        print(f"\n-- BEST MODEL: {best_model[0]} with F1-Score = {best_model[1]['f1']:.4f}")
        print(f"-- Accuracy: {best_model[1]['accuracy']:.4f}, Precision: {best_model[1]['precision']:.4f}, Recall: {best_model[1]['recall']:.4f}")
        
        return best_model[0], best_model[1]['f1']
    
    def save_models(self, path='models/model_a/traditional/'):
        """
        Save all trained models
        """
        os.makedirs(path, exist_ok=True)
        
        for name, model in self.models.items():
            joblib.dump(model, f'{path}/{name}.pkl')
        print(f"\n Models saved to {path}")
    
    def load_models(self, path='models/model_a/traditional/'):
        """
        Load all saved models
        """
        import glob
        
        for model_file in glob.glob(f'{path}/*.pkl'):
            name = os.path.basename(model_file).replace('.pkl', '')
            self.models[name] = joblib.load(model_file)
        print(f"-- Loaded {len(self.models)} models from {path}")



# Main Training Script

if __name__ == "__main__":
    print("-"*70)
    print("Model A --> Answer Verification Training")
    print("-"*70)
    
    # Check if data files exist
    data_dir = 'data/raw/'
    train_path = os.path.join(data_dir, 'train.csv')
    dev_path = os.path.join(data_dir, 'dev.csv')
    
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found!")
        print("Please ensure RACE dataset is in data/raw/ folder")
        exit(1)
    
    # Load data
    print("\n1. Loading RACE dataset...")
    train_df = pd.read_csv(train_path)
    print(f"   Train: {len(train_df)} samples")
    
    if os.path.exists(dev_path):
        val_df = pd.read_csv(dev_path)
        print(f"   Dev: {len(val_df)} samples")
    else:
        print(" dev.csv not found, creating train/validation split...")
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
        print(f"   Train: {len(train_df)} samples, Validation: {len(val_df)} samples")
    
    # Preprocess
    print("\n2. Preprocessing data...")
    preprocessor = RACEPreprocessor(use_tfidf=True, max_vocab_size=3000)
    
    # Use smaller subset for memory and speed
    max_train = 10000   # Reduced for faster training
    max_val = 1000
    
    print(f"   Using {max_train} training samples, {max_val} validation samples")
    print("   Estimated total training time: 15-25 minutes")
    
    X_train, y_train, class_weights = preprocessor.create_option_level_data(train_df, max_samples=max_train)
    X_val, y_val, _ = preprocessor.create_option_level_data(val_df, max_samples=max_val)
    
    print(f"\n   Training features shape: {X_train.shape}")
    print(f"   Validation features shape: {X_val.shape}")
    
    # Class distribution
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    print(f"\n   Class distribution:")
    print(f"     Positive (correct answer): {n_pos} ({n_pos/len(y_train)*100:.1f}%)")
    print(f"     Negative (incorrect): {n_neg} ({n_neg/len(y_train)*100:.1f}%)")
    
    # Initialize trainer
    trainer = ModelATrainer()
    
    # Train models
    print("\n" + "-"*70)
    print("3. Training Models")
    print("-"*70)
    
    # Logistic Regression
    trainer.train_logistic_regression(X_train, y_train, class_weight='balanced')
    trainer.evaluate_model(trainer.models['logistic_regression'], X_val, y_val, 'Logistic Regression')
    
    # Random Forest
    trainer.train_random_forest(X_train, y_train, class_weight='balanced')
    trainer.evaluate_model(trainer.models['random_forest'], X_val, y_val, 'Random Forest')
    
    # XGBoost (needs dense matrix)
    try:
        print("\n Converting to dense for XGBoost...")
        sample_size = min(3000, X_train.shape[0])
        X_train_sample = X_train[:sample_size].toarray() if hasattr(X_train, 'toarray') else X_train[:sample_size]
        y_train_sample = y_train[:sample_size]
        
        trainer.train_xgboost(X_train_sample, y_train_sample)
        X_val_dense = X_val.toarray() if hasattr(X_val, 'toarray') else X_val
        trainer.evaluate_model(trainer.models['xgboost'], X_val_dense, y_val, 'XGBoost')
    except Exception as e:
        print(f" XGBoost skipped: {e}")
    
    # Naive Bayes (keep sparse - this is the fix!)
    trainer.train_naive_bayes(X_train, y_train)  # ← Using sparse X_train, not dense
    trainer.evaluate_model(trainer.models['naive_bayes'], X_val, y_val, 'Naive Bayes')
    
    # SVM (optional - skip if too slow)
    try:
        trainer.train_svm(X_train, y_train, class_weight='balanced')
        trainer.evaluate_model(trainer.models['svm'], X_val, y_val, 'SVM')
    except Exception as e:
        print(f" SVM skipped: {e}")
    
    # Ensemble (if multiple models available)
    if len(trainer.models) >= 2:
        try:
            trainer.train_ensemble(X_train, y_train)
            if 'ensemble_voting' in trainer.models:
                trainer.evaluate_model(trainer.models['ensemble_voting'], X_val, y_val, 'Ensemble Voting')
        except Exception as e:
            print(f" Ensemble skipped: {e}")
    
    # Unsupervised Learning
    print("\n" + "-"*70)
    print("4. Unsupervised & Semi-Supervised Learning")
    print("-"*70)
    
    # K-Means Clustering
    try:
        trainer.unsupervised_kmeans_clustering(X_train, y_train)
    except Exception as e:
        print(f" K-Means skipped: {e}")
    
    # Label Propagation (Semi-Supervised)
    try:
        trainer.semi_supervised_label_propagation(X_train, y_train, labeled_ratio=0.1)
    except Exception as e:
        print(f" Label Propagation skipped: {e}")
    
    # Compare all models
    best_model_name, best_f1 = trainer.compare_all_models()
    
    # Save models
    print("\n" + "-"*70)
    print(" 5. Saving Models")
    print("-"*70)
    trainer.save_models()
    preprocessor.save('models/model_a/traditional/preprocessor.pkl')
    
    print("\n" + "-"*70)
    print("----------- TRAINING COMPLETE! -----------")
    print("-"*70)
    print(f"   Best model: {best_model_name}")
    print(f"   Best F1-Score: {best_f1:.4f}")
    print(f"   Models saved to: models/model_a/traditional/")
    print("-"*70)