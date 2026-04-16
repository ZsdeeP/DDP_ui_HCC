"""
Clinical Outcome Prediction Models - Preoperative Radiomics
Predicts treatment response, survival, tumor characteristics from baseline imaging
Implements all targets identified in preoperative prediction analysis
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, accuracy_score, f1_score
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from datetime import datetime


class OutcomePredictor:
    """Machine learning models for clinical outcome prediction from preoperative imaging"""

    def __init__(self, outcome_type='mRECIST', random_state=42, use_smote=False):
        """
        Args:
            outcome_type: Type of outcome to predict
                # TIER 1 - Treatment Response & Survival
                - 'mRECIST': mRECIST response (HCC-specific) - PRIORITY
                - 'EASL': EASL response
                - 'RECIST': RECIST response
                - 'OS': Overall Survival (will do risk stratification)
                - 'TTP': Time to Progression (will do risk stratification)
                - 'progression': Disease progression status
                # TIER 2 - Tumor Characteristics
                - 'vascular_invasion': Vascular invasion
                - 'cirrhosis': Evidence of cirrhosis
                - 'portal_vein_thrombosis': Portal vein thrombosis
                - 'tumor_nodularity': Tumor nodularity (validation)
                # TIER 2 - Staging
                - 'BCLC': BCLC staging
                - 'CLIP': CLIP score
                - 'AFP_group': AFP group
            random_state: Random seed
            use_smote: Whether to use SMOTE for handling class imbalance
        """
        self.outcome_type = outcome_type
        self.random_state = random_state
        self.use_smote = use_smote
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='constant', fill_value=0)
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.best_model = None
        self.feature_names = None
        self.train_columns = None  # Store training feature columns for alignment

    def prepare_data(self, clinical_df, radiomics_df=None, baseline_features=None):
        """
        Prepare data for outcome prediction with task-specific features

        Args:
            clinical_df: Clinical data DataFrame
            radiomics_df: Task-specific radiomics features (already filtered by region)
            baseline_features: List of baseline clinical features to include (e.g., baseline RECIST, chemo agent)

        Returns:
            X: Feature matrix
            y: Outcome labels
            feature_names: List of feature names
        """
        data = clinical_df.copy()

        # Ensure consistent ID format
        if 'TCIA_ID' in data.columns:
            data['TCIA_ID'] = data['TCIA_ID'].astype(str)

        # Merge task-specific radiomics features
        if radiomics_df is not None:
            radiomics_df = radiomics_df.copy()
            if 'patient_id' in radiomics_df.columns:
                radiomics_df['patient_id'] = radiomics_df['patient_id'].astype(str)
                data = data.merge(radiomics_df, left_on='TCIA_ID', right_on='patient_id', how='inner')

        # Select outcome variable based on type
        outcome_map = {
            # Treatment Response (TIER 1)
            'mRECIST': '1_mRECIST',
            'EASL': '1_EASL',
            'RECIST': '1_RECIST',
            'OS': 'OS',
            'TTP': 'TTP',
            'progression': 'Censored_0_progressed_1',
            # Tumor Characteristics (TIER 2)
            'vascular_invasion': 'Vascular invasion',
            'cirrhosis': 'Evidence_of_cirh',
            'portal_vein_thrombosis': 'Portal Vein Thrombosis',
            'tumor_nodularity': 'tumor_nodul',
            # Staging (TIER 2)
            'BCLC': 'BCLC',
            'CLIP': 'CLIP',
            'AFP_group': 'AFP_group',
            # New outcomes
            'tumor_size': 'Tr_Size',
            'metastasis': 'Metastasis',
            'lymphnodes': 'Lymphnodes',
            'pathology': 'Pathology'
        }

        if self.outcome_type not in outcome_map:
            raise ValueError(f"Unknown outcome type: {self.outcome_type}")

        y_col = outcome_map[self.outcome_type]

        # Check if column exists
        if y_col not in data.columns:
            raise ValueError(f"Outcome column '{y_col}' not found in data")

        # Remove rows with missing outcomes
        data = data.dropna(subset=[y_col])

        # Extract outcome
        y = data[y_col].values

        # Process outcome based on type
        if self.outcome_type == 'EASL':
            # EASL only has SD (3) and PD (4) in this dataset
            # Predict: Stable Disease (1) vs Progressive Disease (0)
            y = (y == 3).astype(int)
            print(f"  Converting EASL to binary: Stable Disease (1) vs Progressive Disease (0)")

        elif self.outcome_type in ['mRECIST', 'RECIST']:
            # Response: 1=CR, 2=PR, 3=SD, 4=PD
            # Binary: Responders (CR+PR) vs Non-responders (SD+PD)
            y = (y <= 2).astype(int)
            print(f"  Converting to binary: Response vs Non-response")

        elif self.outcome_type in ['OS', 'TTP']:
            # For survival, stratify into risk groups (tertiles)
            y_continuous = y.copy()
            tertiles = np.percentile(y, [33.33, 66.67])
            y = np.digitize(y, tertiles)  # 0=Low, 1=Medium, 2=High risk
            print(f"  Converting to risk groups: Low/Medium/High")

        elif self.outcome_type in ['BCLC', 'CLIP']:
            # Merge classes if needed
            if self.outcome_type == 'BCLC':
                # Merge A+B, keep C, merge with D if exists
                mapping = {'Stage-A': 0, 'Stage-B': 0, 'Stage-C': 1, 'Stage-D': 2}
                y = pd.Series(y).map(mapping).fillna(1).astype(int).values
                print(f"  BCLC: Merged to Early(A+B)/C/Advanced(D)")
            else:  # CLIP
                # Binary: 0-2 vs 3-6
                mapping = {'Stage_ 0-2': 0, 'Stage_3': 1, 'Stage_4-6': 1}
                y = pd.Series(y).map(mapping).fillna(0).astype(int).values
                print(f"  CLIP: Binary (0-2 vs 3-6)")

        # Handle categorical encoding for other variables
        elif self.outcome_type == 'AFP_group':
            # Encode AFP groups as binary: <400 vs >=400
            if y.dtype == 'object':
                mapping = {'<400': 0, '>=400': 1}
                y = pd.Series(y).map(mapping).fillna(0).astype(int).values
                print(f"  AFP_group: Encoded as <400 (0) vs >=400 (1)")

        elif self.outcome_type == 'tumor_size':
            # Convert continuous tumor size to tertiles (small/medium/large)
            y_continuous = y.copy()
            tertiles = np.percentile(y[~np.isnan(y)], [33.33, 66.67])
            y = np.digitize(y, tertiles)  # 0=Small, 1=Medium, 2=Large
            print(f"  Tumor size: Converted to size groups (Small: <{tertiles[0]:.1f}cm, Medium: {tertiles[0]:.1f}-{tertiles[1]:.1f}cm, Large: >{tertiles[1]:.1f}cm)")

        elif self.outcome_type == 'pathology':
            # Encode pathology grades (filtering already done in main loop)
            # Encode: Poorly=0, Moderately=1, Well=2
            mapping = {'Poorly differentiated': 0, 'Moderately differentiated': 1, 'Well differentiated': 2}
            y_series = pd.Series(y)
            # Check if there are any invalid values that weren't filtered
            invalid_mask = ~y_series.isin(mapping.keys())
            if invalid_mask.any():
                print(f"  Warning: Found {invalid_mask.sum()} samples with unexpected pathology values. Filtering...")
                data = data[~invalid_mask]
                y = data[y_col].values
                y_series = pd.Series(y)
            y = y_series.map(mapping).values
            print(f"  Pathology: Encoded differentiation grades (Poorly=0, Moderately=1, Well=2). Total: {len(y)} samples")

        elif self.outcome_type in ['metastasis', 'lymphnodes']:
            # Binary encoding (already 0/1 in data)
            y = y.astype(int)
            outcome_name = 'Metastasis' if self.outcome_type == 'metastasis' else 'Lymph nodes'
            print(f"  {outcome_name}: Binary encoding (No=0, Yes=1). Total: {len(y)} samples")

        elif self.outcome_type in ['tumor_nodularity']:
            # Encode as binary
            if y.dtype == 'object':
                y = self.label_encoder.fit_transform(y)

        # Determine which columns to exclude (identity columns and outcome variables)
        # But KEEP baseline_features if specified (e.g., baseline RECIST, chemotherapy)
        exclude_cols = [
            'TCIA_ID', 'patient_id', 'class_number',
            # Outcome variables (don't use future outcomes as predictors)
            'Death_1_StillAliveorLostToFU_0', 'Censored_0_progressed_1',
            'OS', 'TTP', 'Interval_BL', 'Interval_FU',
            # Future response measurements (follow-up)
            '1_RECIST', '2_RECIST', '3_RECIST',
            '1_mRECIST', '2_mRECIST', '3_mRECIST',
            '1_EASL',
            '1_RECIST_FU', '2_RECIST_FU', '3_RECIST_FU',
            '1_EASL_FU',
            '1_mRECIST_FU', '2_mRECIST_FU', '3_mRECIST_FU',
            # Other outcome variables
            'Vascular invasion', 'Evidence_of_cirh', 'Portal Vein Thrombosis',
            'tumor_nodul', 'BCLC', 'CLIP', 'AFP_group',
            'Metastasis', 'Lymphnodes', 'T_involvment', 'Tr_Size', 'Pathology'
        ]

        # Remove the current outcome column from exclusions if it's in there
        if y_col in exclude_cols:
            exclude_cols.remove(y_col)

        # KEEP baseline features if specified (e.g., for treatment response prediction)
        if baseline_features:
            for bf in baseline_features:
                if bf in exclude_cols:
                    exclude_cols.remove(bf)
            print(f"  Including baseline features: {baseline_features}")

        # Select feature columns (radiomics + baseline clinical if specified)
        feature_cols = [col for col in data.columns if col not in exclude_cols]

        # Handle categorical variables
        categorical_cols = data[feature_cols].select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in feature_cols:
                dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
                data = pd.concat([data, dummies], axis=1)
                feature_cols.remove(col)
                feature_cols.extend(dummies.columns.tolist())

        # On first call (training), store the feature columns
        if self.train_columns is None:
            self.train_columns = [col for col in feature_cols if col in data.columns]
            current_cols = self.train_columns
            # Extract features
            X = data[current_cols].values
        else:
            # For validation, align columns to match training
            # Add missing columns with zeros
            for col in self.train_columns:
                if col not in data.columns:
                    data[col] = 0
            # Keep ONLY training columns in the same order
            X = data[self.train_columns].values
            current_cols = self.train_columns

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        self.feature_names = current_cols

        return X, y, current_cols

    def train_models(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train multiple models and compare performance

        Args:
            X_train: Training feature matrix
            y_train: Training outcome labels
            X_val: Validation feature matrix (optional)
            y_val: Validation outcome labels (optional)

        Returns:
            Dictionary of trained models and their scores
        """
        # Impute missing values first
        X_train_imputed = self.imputer.fit_transform(X_train)

        # Scale features using training set statistics
        X_train_scaled = self.scaler.fit_transform(X_train_imputed)

        # Scale validation set if provided
        if X_val is not None and y_val is not None:
            X_val_imputed = self.imputer.transform(X_val)
            X_val_scaled = self.scaler.transform(X_val_imputed)
            has_val = True
        else:
            has_val = False

        # Apply SMOTE if requested and data is appropriate
        if self.use_smote and len(np.unique(y_train)) == 2:
            # Only apply SMOTE for binary classification
            try:
                smote = SMOTE(random_state=self.random_state, k_neighbors=min(5, np.bincount(y_train).min() - 1))
                X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
                print(f"  SMOTE applied. New class distribution: {np.bincount(y_train)}")
            except Exception as e:
                print(f"  SMOTE failed: {e}. Continuing without SMOTE.")

        # Define models - 5 different methods for comparison
        model_configs = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, class_weight='balanced',
                C=0.1, solver='liblinear',
                random_state=self.random_state
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_split=10,
                min_samples_leaf=5, class_weight='balanced',
                random_state=self.random_state, n_jobs=-1
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5,
                subsample=0.8, colsample_bytree=0.8,
                random_state=self.random_state, n_jobs=-1,
                eval_metric='logloss'
            ),
            'SVM (RBF)': SVC(
                kernel='rbf', C=1.0, gamma='scale',
                class_weight='balanced', probability=True,
                random_state=self.random_state
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=500,
                alpha=0.01, learning_rate='adaptive',
                random_state=self.random_state
            )
        }

        results = {}

        print(f"\nTraining models for {self.outcome_type} prediction...")
        print(f"Training samples: {len(X_train)}")
        if has_val:
            print(f"Validation samples: {len(X_val)}")
        print(f"Class distribution - Train: {np.bincount(y_train)}")
        if has_val:
            print(f"Class distribution - Val: {np.bincount(y_val)}")

        best_val_auc = 0
        best_train_auc = 0

        for name, model in model_configs.items():
            print(f"\n{name}:")

            # Train
            model.fit(X_train_scaled, y_train)

            # Evaluate on training set
            y_train_pred = model.predict(X_train_scaled)
            y_train_proba = model.predict_proba(X_train_scaled)

            train_accuracy = accuracy_score(y_train, y_train_pred)

            # Handle AUC for binary vs multiclass
            n_classes = len(np.unique(y_train))
            if n_classes == 2:
                train_auc = roc_auc_score(y_train, y_train_proba[:, 1])
            else:
                # Multiclass: use one-vs-rest strategy
                train_auc = roc_auc_score(y_train, y_train_proba, multi_class='ovr', average='weighted')

            train_f1 = f1_score(y_train, y_train_pred, average='weighted')

            print(f"  Train Accuracy: {train_accuracy:.4f}")
            print(f"  Train AUC: {train_auc:.4f}")
            print(f"  Train F1-Score: {train_f1:.4f}")

            # Store model and results
            self.models[name] = model
            results[name] = {
                'model': model,
                'train_accuracy': train_accuracy,
                'train_auc': train_auc,
                'train_f1': train_f1,
                'y_train_pred': y_train_pred,
                'y_train_proba': y_train_proba,
                'y_train': y_train
            }

            # Evaluate on validation set if provided
            if has_val:
                y_val_pred = model.predict(X_val_scaled)
                y_val_proba = model.predict_proba(X_val_scaled)

                val_accuracy = accuracy_score(y_val, y_val_pred)

                # Handle AUC for binary vs multiclass
                n_classes_val = len(np.unique(y_val))
                if n_classes_val == 2:
                    val_auc = roc_auc_score(y_val, y_val_proba[:, 1])
                else:
                    # Multiclass: use one-vs-rest strategy
                    val_auc = roc_auc_score(y_val, y_val_proba, multi_class='ovr', average='weighted')

                val_f1 = f1_score(y_val, y_val_pred, average='weighted')

                print(f"  Val Accuracy: {val_accuracy:.4f}")
                print(f"  Val AUC: {val_auc:.4f}")
                print(f"  Val F1-Score: {val_f1:.4f}")

                # Add validation metrics to results
                results[name].update({
                    'val_accuracy': val_accuracy,
                    'val_auc': val_auc,
                    'val_f1': val_f1,
                    'y_val_pred': y_val_pred,
                    'y_val_proba': y_val_proba,
                    'y_val': y_val
                })

                # Track best model based on validation AUC
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_train_auc = train_auc
                    self.best_model = model
                    self.best_model_name = name
            else:
                # If no validation set, use training AUC
                if train_auc > best_train_auc:
                    best_train_auc = train_auc
                    self.best_model = model
                    self.best_model_name = name

        if has_val:
            print(f"\nBest model: {self.best_model_name} (Val AUC: {best_val_auc:.4f}, Train AUC: {best_train_auc:.4f})")
        else:
            print(f"\nBest model: {self.best_model_name} (Train AUC: {best_train_auc:.4f})")

        return results

    def plot_roc_curves(self, results, save_path=None, set_name='validation'):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))

        # Determine which set to plot
        use_val = 'y_val' in list(results.values())[0]
        y_key = 'y_val' if use_val else 'y_train'
        proba_key = 'y_val_proba' if use_val else 'y_train_proba'
        auc_key = 'val_auc' if use_val else 'train_auc'

        for name, result in results.items():
            fpr, tpr, _ = roc_curve(result[y_key], result[proba_key])
            auc = result[auc_key]
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2)

        plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        set_label = 'Validation' if use_val else 'Training'
        plt.title(f'ROC Curves ({set_label} Set) - {self.outcome_type.capitalize()} Prediction', fontsize=14)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred, save_path=None, set_name='validation'):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title(f'Confusion Matrix ({set_name.capitalize()} Set) - {self.outcome_type.capitalize()} Prediction', fontsize=14)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def get_feature_importance(self, top_n=20):
        """Get feature importance from best model"""
        if self.best_model is None:
            raise ValueError("No model trained yet")

        if hasattr(self.best_model, 'feature_importances_'):
            # Tree-based models
            importances = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            # Linear models
            importances = np.abs(self.best_model.coef_[0])
        else:
            print("Feature importance not available for this model")
            return None

        # Sort features by importance
        indices = np.argsort(importances)[::-1][:top_n]

        return [(self.feature_names[i], importances[i]) for i in indices]

    def plot_feature_importance(self, top_n=20, save_path=None):
        """Plot feature importance"""
        feature_imp = self.get_feature_importance(top_n)

        if feature_imp is None:
            return

        features, importances = zip(*feature_imp)

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(features)), importances, color='steelblue')
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance', fontsize=12)
        plt.title(f'Top {top_n} Features - {self.outcome_type.capitalize()} Prediction', fontsize=14)
        plt.grid(axis='x', alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def save_model(self, filepath):
        """Save the best model"""
        if self.best_model is None:
            raise ValueError("No model trained yet")

        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'outcome_type': self.outcome_type,
            'model_name': self.best_model_name
        }

        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    @staticmethod
    def load_model(filepath):
        """Load a saved model"""
        model_data = joblib.load(filepath)

        predictor = OutcomePredictor(outcome_type=model_data['outcome_type'])
        predictor.best_model = model_data['model']
        predictor.scaler = model_data['scaler']
        predictor.feature_names = model_data['feature_names']
        predictor.best_model_name = model_data['model_name']

        return predictor


def create_stratified_split(clinical_df, radiomics_df, outcome_col, test_size=0.2, random_state=42):
    """
    Create stratified train/val split based on clinical outcome

    Args:
        clinical_df: Clinical data
        radiomics_df: Radiomics features
        outcome_col: Column name for stratification
        test_size: Validation set size
        random_state: Random seed

    Returns:
        train_radiomics, val_radiomics, train_clinical, val_clinical
    """
    # Merge to get complete cases
    data = clinical_df.copy()
    data['TCIA_ID'] = data['TCIA_ID'].astype(str)

    if radiomics_df is not None and 'patient_id' in radiomics_df.columns:
        radiomics_df = radiomics_df.copy()
        radiomics_df['patient_id'] = radiomics_df['patient_id'].astype(str)
        data = data.merge(radiomics_df[['patient_id']], left_on='TCIA_ID', right_on='patient_id', how='inner')

    # Remove missing outcomes
    data = data.dropna(subset=[outcome_col])
    patient_ids = data['TCIA_ID'].values
    outcomes = data[outcome_col].values

    # Stratified split
    train_ids, val_ids = train_test_split(
        patient_ids, test_size=test_size,
        stratify=outcomes, random_state=random_state
    )

    # Split radiomics
    if radiomics_df is not None:
        train_radiomics = radiomics_df[radiomics_df['patient_id'].isin(train_ids)].copy()
        val_radiomics = radiomics_df[radiomics_df['patient_id'].isin(val_ids)].copy()
    else:
        train_radiomics = None
        val_radiomics = None

    # Split clinical (keep all for reference)
    train_clinical = clinical_df[clinical_df['TCIA_ID'].isin(train_ids)].copy()
    val_clinical = clinical_df[clinical_df['TCIA_ID'].isin(val_ids)].copy()

    print(f"\nStratified split on '{outcome_col}':")
    print(f"  Train: {len(train_ids)} patients")
    print(f"  Val: {len(val_ids)} patients")

    return train_radiomics, val_radiomics, train_clinical, val_clinical


def run_outcome_prediction(clinical_csv, radiomics_dir='results/radiomics', output_dir='results/outcome_prediction'):
    """
    Run comprehensive outcome prediction for all preoperative targets
    Uses task-specific radiomics regions for each clinical outcome

    Args:
        clinical_csv: Path to clinical data
        radiomics_dir: Directory containing radiomics features
        output_dir: Directory to save results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    clinical_df = pd.read_csv(clinical_csv)

    # Load radiomics by region, combining train/val splits
    # The train/val split was for segmentation task, not clinical prediction
    radiomics_by_region = {}
    region_names = ['tumor', 'liver', 'bloodvessels', 'abdominalaorta']

    print("\nLoading radiomics features by anatomical region...")
    for region in region_names:
        train_file = Path(radiomics_dir) / f'radiomics_features_{region}_train.csv'
        val_file = Path(radiomics_dir) / f'radiomics_features_{region}_val.csv'

        dfs = []
        if train_file.exists():
            dfs.append(pd.read_csv(train_file))
        if val_file.exists():
            dfs.append(pd.read_csv(val_file))

        if dfs:
            # Combine train and val
            region_df = pd.concat(dfs, ignore_index=True)
            # Remove duplicate columns if any
            region_df = region_df.loc[:, ~region_df.columns.duplicated()]
            radiomics_by_region[region] = region_df
            print(f"  {region}: {region_df.shape[0]} patients, {region_df.shape[1]-2} features")  # -2 for patient_id and class_number

    if not radiomics_by_region:
        print("ERROR: No radiomics features found!")
        return

    # Create results summary list
    all_results = []

    # Define all prediction targets with their required radiomics regions and baseline clinical features
    outcome_configs = {
        # TIER 1: Treatment Response (need tumor radiomics + baseline measurements + treatment info)
        'mRECIST': {
            'col': '1_mRECIST',
            'smote': True,
            'tier': 1,
            'priority': 1,
            'regions': ['tumor'],  # Tumor radiomics for treatment response
            'baseline_features': ['1_mRECIST_BL', 'chemotherapy'],  # Baseline mRECIST and treatment agent
            'description': 'Predict treatment response (mRECIST) from baseline tumor radiomics + baseline mRECIST + chemo agent'
        },
        'RECIST': {
            'col': '1_RECIST',
            'smote': True,
            'tier': 1,
            'priority': 2,
            'regions': ['tumor'],
            'baseline_features': ['1_RECIST_BL', 'chemotherapy'],  # Baseline RECIST and treatment agent
            'description': 'Predict treatment response (RECIST) from baseline tumor radiomics + baseline RECIST + chemo agent'
        },
        'EASL': {
            'col': '1_EASL',
            'smote': True,
            'tier': 1,
            'priority': 3,
            'regions': ['tumor'],
            'baseline_features': ['1_EASL_BL', 'chemotherapy'],
            'description': 'Predict disease control (Stable Disease vs Progressive Disease) from tumor radiomics + baseline EASL + chemo'
        },
        'progression': {
            'col': 'Censored_0_progressed_1',
            'smote': False,
            'tier': 1,
            'priority': 4,
            'regions': ['tumor'],
            'baseline_features': ['BCLC', 'CLIP'],  # Staging info
            'description': 'Predict disease progression from tumor radiomics + staging'
        },

        # TIER 2: Tumor Characteristics (tumor radiomics only)
        'tumor_nodularity': {
            'col': 'tumor_nodul',
            'smote': False,
            'tier': 2,
            'priority': 5,
            'regions': ['tumor'],
            'baseline_features': [],
            'description': 'Predict tumor nodularity (single vs multi-nodular) from tumor radiomics'
        },
        'BCLC': {
            'col': 'BCLC',
            'smote': True,
            'tier': 2,
            'priority': 6,
            'regions': ['tumor'],
            'baseline_features': [],
            'description': 'Predict BCLC staging from tumor radiomics'
        },
        'CLIP': {
            'col': 'CLIP',
            'smote': True,
            'tier': 2,
            'priority': 7,
            'regions': ['tumor'],
            'baseline_features': [],
            'description': 'Predict CLIP score from tumor radiomics'
        },

        # TIER 2: Vascular Features (vessel radiomics)
        'vascular_invasion': {
            'col': 'Vascular invasion',
            'smote': True,
            'tier': 2,
            'priority': 8,
            'regions': ['bloodvessels'],  # Vessel radiomics for vascular invasion
            'baseline_features': [],
            'description': 'Predict vascular invasion from blood vessel radiomics'
        },
        'portal_vein_thrombosis': {
            'col': 'Portal Vein Thrombosis',
            'smote': True,
            'tier': 2,
            'priority': 9,
            'regions': ['bloodvessels', 'abdominalaorta'],  # Vessel + aorta for thrombosis
            'baseline_features': [],
            'description': 'Predict portal vein thrombosis from vessel radiomics'
        },

        # TIER 2: Liver Features (liver radiomics)
        'cirrhosis': {
            'col': 'Evidence_of_cirh',
            'smote': True,
            'tier': 2,
            'priority': 10,
            'regions': ['liver'],  # Liver radiomics for cirrhosis
            'baseline_features': [],
            'description': 'Predict cirrhosis from liver parenchyma radiomics'
        },

        # Other
        'AFP_group': {
            'col': 'AFP_group',
            'smote': True,
            'tier': 2,
            'priority': 11,
            'regions': ['tumor', 'liver'],  # Tumor + liver for AFP
            'baseline_features': [],
            'description': 'Predict AFP level group from tumor + liver radiomics'
        },

        # TIER 2: Tumor Measurements (tumor radiomics)
        'tumor_size': {
            'col': 'Tr_Size',
            'smote': True,  # Use SMOTE for imbalanced size groups
            'tier': 2,
            'priority': 12,
            'regions': ['tumor'],
            'baseline_features': [],
            'description': 'Predict tumor size from tumor radiomics (converted to size groups)',
            'task_type': 'classification'  # Will convert continuous size to categories
        },

        # TIER 2: Metastatic Features
        'metastasis': {
            'col': 'Metastasis',
            'smote': True,
            'tier': 2,
            'priority': 13,
            'regions': ['tumor', 'liver'],  # Tumor + liver for metastasis detection
            'baseline_features': [],
            'description': 'Predict presence of metastasis from tumor + liver radiomics'
        },
        'lymphnodes': {
            'col': 'Lymphnodes',
            'smote': True,
            'tier': 2,
            'priority': 14,
            'regions': ['tumor', 'bloodvessels'],  # Tumor + vessels for lymph node involvement
            'baseline_features': [],
            'description': 'Predict lymph node involvement from tumor + vessel radiomics'
        },

        # TIER 2: Pathology (tumor radiomics)
        'pathology': {
            'col': 'Pathology',
            'smote': True,  # Use SMOTE for imbalanced differentiation grades
            'tier': 2,
            'priority': 15,
            'regions': ['tumor'],
            'baseline_features': [],
            'description': 'Predict tumor differentiation grade from tumor radiomics',
            'filter_values': ['NOT STATED', 'No biopsy', 'Moderately-poorly differentiated', 'Well-moderately differentiated']  # Exclude these values
        },
    }

    # Sort by priority
    outcomes = sorted(outcome_configs.keys(), key=lambda x: outcome_configs[x]['priority'])

    for outcome_type in outcomes:
        print("\n" + "="*80)
        config = outcome_configs[outcome_type]
        print(f"TIER {config['tier']} - {outcome_type.upper().replace('_', ' ')} PREDICTION")
        print(f"Description: {config['description']}")
        print("="*80)

        try:
            outcome_col = config['col']
            use_smote = config['smote']
            required_regions = config['regions']
            baseline_features = config.get('baseline_features', [])

            # Check if outcome column exists
            if outcome_col not in clinical_df.columns:
                print(f"  Column '{outcome_col}' not found. Skipping.")
                continue

            # Special handling for pathology: filter out unwanted values before split
            clinical_df_filtered = clinical_df.copy()
            if 'filter_values' in config:
                filter_values = config['filter_values']
                before_count = len(clinical_df_filtered)
                clinical_df_filtered = clinical_df_filtered[~clinical_df_filtered[outcome_col].isin(filter_values)]
                after_count = len(clinical_df_filtered)
                print(f"Filtered outcome: removed {before_count - after_count} samples with values {filter_values}")
                print(f"Remaining samples: {after_count}")

                if after_count < 10:
                    print(f"  Insufficient samples after filtering ({after_count}). Skipping.")
                    continue

            # Merge task-specific radiomics regions
            task_radiomics = None
            for region in required_regions:
                if region not in radiomics_by_region:
                    print(f"  Required region '{region}' not found. Skipping.")
                    continue

                region_df = radiomics_by_region[region].copy()
                # Add region prefix to avoid column name conflicts
                feature_cols = [c for c in region_df.columns if c not in ['patient_id', 'class_number']]
                region_df = region_df.rename(columns={c: f'{region}_{c}' for c in feature_cols})

                if task_radiomics is None:
                    task_radiomics = region_df
                else:
                    task_radiomics = task_radiomics.merge(region_df, on='patient_id', how='inner')

            if task_radiomics is None:
                print(f"  No radiomics data available for required regions. Skipping.")
                continue

            print(f"Using radiomics from: {', '.join(required_regions)}")
            print(f"Combined: {task_radiomics.shape[0]} patients, {task_radiomics.shape[1]-1} features")

            # Special preprocessing for tumor_size: convert to groups before stratification
            if outcome_type == 'tumor_size':
                valid_sizes = clinical_df_filtered[outcome_col].dropna()
                if len(valid_sizes) < 30:
                    print(f"  Insufficient samples with tumor size ({len(valid_sizes)}). Skipping.")
                    continue

                # Convert to tertiles
                tertiles = np.percentile(valid_sizes, [33.33, 66.67])
                clinical_df_filtered[f'{outcome_col}_group'] = clinical_df_filtered[outcome_col].apply(
                    lambda x: np.digitize(x, tertiles) if pd.notna(x) else np.nan
                )
                # Use the grouped version for stratification
                stratify_col = f'{outcome_col}_group'
                print(f"  Converted tumor size to tertiles: Small<{tertiles[0]:.1f}, Medium:{tertiles[0]:.1f}-{tertiles[1]:.1f}, Large>{tertiles[1]:.1f}")
            else:
                stratify_col = outcome_col

            # Create stratified split based on this outcome (use filtered clinical data if applicable)
            train_radiomics, val_radiomics, train_clinical, val_clinical = create_stratified_split(
                clinical_df_filtered, task_radiomics, stratify_col, test_size=0.2, random_state=42
            )

            # Initialize predictor
            predictor = OutcomePredictor(outcome_type=outcome_type, use_smote=use_smote)

            # Prepare training data with baseline features
            try:
                X_train, y_train, feature_names = predictor.prepare_data(
                    train_clinical, train_radiomics, baseline_features=baseline_features
                )
                print(f"Train Dataset: {X_train.shape[0]} patients, {X_train.shape[1]} features")
                print(f"Train Class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
            except Exception as e:
                print(f"  Error preparing training data: {e}")
                continue

            # Prepare validation data
            try:
                X_val, y_val, _ = predictor.prepare_data(
                    val_clinical, val_radiomics, baseline_features=baseline_features
                )
                print(f"Val Dataset: {X_val.shape[0]} patients, {X_val.shape[1]} features")
                print(f"Val Class distribution: {dict(zip(*np.unique(y_val, return_counts=True)))}")
                has_val = True
            except Exception as e:
                print(f"  Error preparing validation data: {e}")
                X_val, y_val = None, None
                has_val = False

            # Skip if insufficient data
            if len(y_train) < 10:
                print(f"  Insufficient training samples ({len(y_train)}). Skipping.")
                continue

            # Train models
            results = predictor.train_models(X_train, y_train, X_val, y_val)

            # Save metrics to summary
            for model_name, model_results in results.items():
                summary_row = {
                    'tier': config['tier'],
                    'priority': config['priority'],
                    'outcome': outcome_type,
                    'outcome_col': outcome_col,
                    'model': model_name,
                    'n_train': len(y_train),
                    'n_val': len(y_val) if has_val else 0,
                    'train_accuracy': model_results['train_accuracy'],
                    'train_auc': model_results.get('train_auc', 0),
                    'train_f1': model_results['train_f1'],
                    'used_smote': use_smote
                }
                if has_val:
                    summary_row.update({
                        'val_accuracy': model_results['val_accuracy'],
                        'val_auc': model_results.get('val_auc', 0),
                        'val_f1': model_results['val_f1']
                    })
                all_results.append(summary_row)

            # Create outcome-specific directory
            outcome_dir = output_path / outcome_type
            outcome_dir.mkdir(exist_ok=True)

            # Plot results
            try:
                predictor.plot_roc_curves(results,
                    save_path=outcome_dir / 'roc_curves.png')
            except Exception as e:
                print(f"  Warning: Could not plot ROC curves: {e}")

            # Plot confusion matrix for best model
            try:
                if has_val:
                    y_pred = results[predictor.best_model_name]['y_val_pred']
                    y_true = results[predictor.best_model_name]['y_val']
                    set_name = 'validation'
                else:
                    y_pred = results[predictor.best_model_name]['y_train_pred']
                    y_true = results[predictor.best_model_name]['y_train']
                    set_name = 'training'

                predictor.plot_confusion_matrix(y_true, y_pred,
                    save_path=outcome_dir / 'confusion_matrix.png',
                    set_name=set_name)
            except Exception as e:
                print(f"  Warning: Could not plot confusion matrix: {e}")

            # Plot feature importance
            try:
                predictor.plot_feature_importance(top_n=20,
                    save_path=outcome_dir / 'feature_importance.png')
            except Exception as e:
                print(f"  Warning: Could not plot feature importance: {e}")

            # Save best model
            try:
                predictor.save_model(outcome_dir / 'best_model.pkl')
            except Exception as e:
                print(f"  Warning: Could not save model: {e}")

            # Generate report
            report_lines = [
                f"Outcome Prediction Report: {outcome_type}",
                "=" * 80,
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "Data Summary:",
                f"  Training samples: {len(y_train)}",
                f"  Validation samples: {len(y_val) if has_val else 0}",
                f"  Number of features: {X_train.shape[1]}",
                f"  SMOTE applied: {use_smote}",
                "",
                "Best Model:",
                f"  Model: {predictor.best_model_name}",
                f"  Train AUC: {results[predictor.best_model_name].get('train_auc', 0):.4f}",
            ]
            if has_val:
                report_lines.append(f"  Val AUC: {results[predictor.best_model_name].get('val_auc', 0):.4f}")

            with open(outcome_dir / 'report.txt', 'w') as f:
                f.write('\n'.join(report_lines))

        except Exception as e:
            print(f"Error in {outcome_type} prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Save comprehensive results summary
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values(['tier', 'priority', 'model'])
        results_df.to_csv(output_path / 'all_results_summary.csv', index=False)

        print("\n" + "="*80)
        print("COMPREHENSIVE RESULTS SUMMARY")
        print("="*80)

        # Display by tier
        for tier in sorted(results_df['tier'].unique()):
            tier_df = results_df[results_df['tier'] == tier]
            print(f"\nTIER {tier}:")
            print(tier_df[['outcome', 'model', 'train_auc', 'val_auc', 'val_accuracy']].to_string(index=False))

        # Find best models per outcome
        print("\n" + "="*80)
        print("BEST MODELS PER OUTCOME")
        print("="*80)
        best_models = results_df.loc[results_df.groupby('outcome')['val_auc'].idxmax()]
        print(best_models[['outcome', 'model', 'train_auc', 'val_auc', 'val_accuracy', 'used_smote']].to_string(index=False))

        # Save best models summary
        best_models.to_csv(output_path / 'best_models_summary.csv', index=False)

    print("\n" + "="*80)
    print("OUTCOME PREDICTION COMPLETE!")
    print(f"Results saved to: {output_path}")
    print("="*80)
    print("\nGenerated outputs:")
    print("  - all_results_summary.csv: Complete results for all models")
    print("  - best_models_summary.csv: Best performing model per outcome")
    print("  - <outcome>/: Individual directories with plots and models")
    print("="*80)


if __name__ == "__main__":
    import sys

    # Default paths
    clinical_csv = "/data/data/HCC/agent_work/clinical_data.csv"
    radiomics_dir = "/data/data/HCC/agent_work/results/radiomics"
    output_dir = "/data/data/HCC/agent_work/results/outcome_prediction"

    # Allow command line override
    if len(sys.argv) > 1:
        clinical_csv = sys.argv[1]
    if len(sys.argv) > 2:
        radiomics_dir = sys.argv[2]
    if len(sys.argv) > 3:
        output_dir = sys.argv[3]

    print("="*80)
    print("PREOPERATIVE RADIOMICS - OUTCOME PREDICTION")
    print("="*80)
    print(f"Clinical data: {clinical_csv}")
    print(f"Radiomics dir: {radiomics_dir}")
    print(f"Output dir: {output_dir}")
    print("="*80)

    run_outcome_prediction(clinical_csv, radiomics_dir, output_dir)
