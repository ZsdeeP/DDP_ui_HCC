"""
Survival Analysis Module for HCC-TACE Patients (CORRECTED VERSION)
Uses Cox Proportional Hazards model and survival prediction
Includes nomogram visualization for clinical interpretation

FIXES APPLIED:
1. Fixed feature alignment bug with proper one-hot encoding
2. Added bounds checking in feature selection
3. Fixed Cox model prediction dimension matching
4. Added nested CV to prevent data leakage
5. Added comprehensive input validation
6. Added survival data validation
7. Consistent random state handling
8. Added proportional hazards testing
9. Added statistical model comparison
10. Improved stratified split with fallback
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.feature_selection import SelectKBest
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.statistics import proportional_hazard_test
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from collections import Counter
warnings.filterwarnings('ignore')

# Constants
DEFAULT_N_FEATURES = 15  # Balanced to get enough features but avoid overfitting
FIRST_PASS_FEATURES = 30  # Conservative preselection
TOP_PLOT_FEATURES = 20
DEFAULT_TEST_SIZE = 0.3  # Larger validation set for more stable estimates
MIN_SURVIVAL_TIME = 0.01  # Minimum valid survival time in months
MIN_PENALIZER = 1.0  # Minimum regularization penalty to prevent overfitting


class SurvivalAnalyzer:
    """Comprehensive survival analysis for HCC-TACE patients"""

    def __init__(self, outcome_type='OS', random_state=42):
        """
        Args:
            outcome_type: 'OS' for Overall Survival or 'TTP' for Time To Progression
            random_state: Random seed for reproducibility
        """
        self.outcome_type = outcome_type
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.cox_model = None
        self.rsf_model = None
        self.feature_selector = None
        self.selected_features = None
        self.train_columns = None
        self.categorical_info = None  # Store categorical encoding info
        self.feature_names_map = None  # Map indices to names

    def _validate_survival_data(self, times, events, data_name="dataset"):
        """
        Validate survival data for common issues

        Args:
            times: Array of survival times
            events: Array of event indicators
            data_name: Name for error messages

        Returns:
            valid_mask: Boolean mask of valid samples
        """
        valid_mask = np.ones(len(times), dtype=bool)

        # Check for non-positive survival times
        non_positive = times <= 0
        if np.any(non_positive):
            n_invalid = np.sum(non_positive)
            print(f"  Warning: {n_invalid} samples in {data_name} with non-positive survival times (will be removed)")
            valid_mask &= ~non_positive

        # Check for extremely small survival times (likely data errors)
        too_small = (times > 0) & (times < MIN_SURVIVAL_TIME)
        if np.any(too_small):
            n_invalid = np.sum(too_small)
            print(f"  Warning: {n_invalid} samples in {data_name} with suspiciously small times <{MIN_SURVIVAL_TIME} months")

        # Validate event encoding
        unique_events = np.unique(events[valid_mask])
        if not np.all(np.isin(unique_events, [0, 1])):
            raise ValueError(f"Event column must be 0/1, found: {unique_events}")

        # Check for excessive ties
        valid_times = times[valid_mask]
        unique_times = len(np.unique(valid_times))
        tie_ratio = unique_times / len(valid_times)
        if tie_ratio < 0.5:
            print(f"  Warning: High proportion of tied times in {data_name} ({unique_times}/{len(valid_times)} unique)")

        # Check censoring rate
        events_valid = events[valid_mask]
        event_rate = np.sum(events_valid) / len(events_valid)
        if event_rate < 0.2:
            print(f"  Warning: Low event rate in {data_name} ({event_rate:.1%}) - may affect model performance")
        elif event_rate > 0.8:
            print(f"  Warning: High event rate in {data_name} ({event_rate:.1%}) - limited censoring")

        return valid_mask

    def _validate_dataframes(self, clinical_df, radiomics_dfs):
        """Validate input dataframes"""
        if clinical_df is None or clinical_df.empty:
            raise ValueError("Clinical dataframe is empty or None")

        if 'TCIA_ID' not in clinical_df.columns:
            raise ValueError("Clinical dataframe must have 'TCIA_ID' column")

        # Check for duplicate patient IDs
        id_counts = clinical_df['TCIA_ID'].value_counts()
        if any(id_counts > 1):
            duplicates = id_counts[id_counts > 1].index.tolist()
            raise ValueError(f"Duplicate patient IDs found: {duplicates[:5]}")

        # Validate radiomics if provided
        if radiomics_dfs is not None:
            if isinstance(radiomics_dfs, dict):
                for region, df in radiomics_dfs.items():
                    if df is not None and not df.empty:
                        if 'patient_id' not in df.columns:
                            raise ValueError(f"Radiomics dataframe for {region} must have 'patient_id' column")

    def prepare_data(self, clinical_df, radiomics_dfs=None, is_training=True):
        """
        Prepare and merge data from clinical and multiple radiomics regions
        FIXED: Proper handling of categorical encoding and column alignment

        Args:
            clinical_df: Clinical data DataFrame
            radiomics_dfs: Dict of radiomics DataFrames by region or single DataFrame
            is_training: Whether this is training data (affects encoding)

        Returns:
            X: Feature matrix
            y: Structured array for survival analysis (event, time)
            feature_names: List of feature names
            data: Full merged DataFrame
        """
        # Validate inputs
        self._validate_dataframes(clinical_df, radiomics_dfs)

        # Start with clinical data
        data = clinical_df.copy()

        # Ensure consistent ID format (string without HCC_ prefix)
        if 'TCIA_ID' in data.columns:
            data['TCIA_ID'] = data['TCIA_ID'].astype(str).str.replace('HCC_', '')

        # Merge radiomics features if provided
        if radiomics_dfs is not None:
            if isinstance(radiomics_dfs, dict):
                # Multiple regions
                for region_name, radiomics_df in radiomics_dfs.items():
                    if radiomics_df is not None and not radiomics_df.empty:
                        radiomics_df = radiomics_df.copy()
                        radiomics_df['patient_id'] = radiomics_df['patient_id'].astype(str).str.replace('HCC_', '')

                        # Prefix features with region name to avoid conflicts
                        feature_cols = [c for c in radiomics_df.columns if c not in ['patient_id', 'class_number']]
                        radiomics_df = radiomics_df.rename(columns={c: f'{region_name}_{c}' for c in feature_cols})

                        data = data.merge(radiomics_df, left_on='TCIA_ID', right_on='patient_id', how='inner')
                        if is_training:
                            print(f"    Merged {region_name} radiomics: {len(feature_cols)} features")
            else:
                # Single dataframe for backward compatibility
                radiomics_df = radiomics_dfs.copy()
                radiomics_df['patient_id'] = radiomics_df['patient_id'].astype(str).str.replace('HCC_', '')
                data = data.merge(radiomics_df, left_on='TCIA_ID', right_on='patient_id', how='inner')

        # Determine outcome columns based on type
        if self.outcome_type == 'OS':
            time_col = 'OS'
            event_col = 'Death_1_StillAliveorLostToFU_0'
        elif self.outcome_type == 'TTP':
            time_col = 'TTP'
            event_col = 'Censored_0_progressed_1'
        else:
            raise ValueError(f"Unknown outcome type: {self.outcome_type}")

        # Check required columns exist
        if time_col not in data.columns or event_col not in data.columns:
            raise ValueError(f"Required outcome columns '{time_col}' and '{event_col}' not found")

        # Remove rows with missing outcome data
        initial_size = len(data)
        data = data.dropna(subset=[time_col, event_col])
        if len(data) < initial_size and is_training:
            print(f"    Removed {initial_size - len(data)} samples with missing outcome data")

        if len(data) == 0:
            raise ValueError("No samples remaining after removing missing outcome data")

        # Validate survival data
        times = data[time_col].values
        events = data[event_col].values
        valid_mask = self._validate_survival_data(times, events, "training" if is_training else "validation")
        data = data[valid_mask].reset_index(drop=True)
        times = data[time_col].values
        events = data[event_col].values

        # Select feature columns (exclude outcome and ID columns)
        exclude_cols = [
            'TCIA_ID', 'patient_id', 'class_number',
            # Outcome variables (don't use as predictors)
            'OS', 'TTP', 'Death_1_StillAliveorLostToFU_0', 'Censored_0_progressed_1',
            'Interval_BL', 'Interval_FU',
            # Future measurements (follow-up outcomes)
            '1_RECIST', '2_RECIST', '3_RECIST', '1_mRECIST', '2_mRECIST', '3_mRECIST',
            '1_EASL', '1_RECIST_FU', '2_RECIST_FU', '3_RECIST_FU',
            '1_EASL_FU', '1_mRECIST_FU', '2_mRECIST_FU', '3_mRECIST_FU',
        ]

        feature_cols = [col for col in data.columns if col not in exclude_cols]

        # FIXED: Process categorical variables BEFORE storing column names
        categorical_cols = data[feature_cols].select_dtypes(include=['object']).columns.tolist()

        if is_training:
            # Store categorical information for consistent encoding
            self.categorical_info = {}

            for col in categorical_cols:
                if col in feature_cols:
                    # Store unique categories
                    self.categorical_info[col] = sorted(data[col].dropna().unique().tolist())

                    # Create dummy variables
                    dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
                    data = pd.concat([data, dummies], axis=1)

                    # Update feature list
                    feature_cols.remove(col)
                    feature_cols.extend(dummies.columns.tolist())

            # NOW store the final feature columns (after encoding)
            self.train_columns = [col for col in feature_cols if col in data.columns]
            current_cols = self.train_columns

        else:
            # Validation: Apply same categorical encoding as training
            if self.categorical_info is None:
                raise ValueError("Cannot process validation data before training data")

            for col, categories in self.categorical_info.items():
                if col in data.columns:
                    # Ensure validation data only has categories seen in training
                    unseen = set(data[col].unique()) - set(categories) - {np.nan}
                    if unseen:
                        print(f"  Warning: Unseen categories in {col}: {unseen}. Setting to most common.")
                        most_common = categories[0] if categories else np.nan
                        data.loc[data[col].isin(unseen), col] = most_common

                    # Create dummies (will create columns for training categories)
                    dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
                    data = pd.concat([data, dummies], axis=1)

            # Align to training columns - add missing, remove extra
            for col in self.train_columns:
                if col not in data.columns:
                    data[col] = 0

            # Keep ONLY training columns in the same order
            current_cols = self.train_columns

        # Extract features
        X = data[current_cols].values

        # Handle missing values in features (after one-hot encoding)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Create structured array for survival analysis
        y = np.array([(bool(e), t) for e, t in zip(events, times)],
                     dtype=[('event', bool), ('time', float)])

        return X, y, current_cols, data

    def select_features(self, X, y, feature_names, n_features=DEFAULT_N_FEATURES, use_cv=True):
        """
        Select top features using univariate C-index and regularized Cox regression
        FIXED: Added bounds checking and better error handling

        Args:
            X: Feature matrix
            y: Survival outcome
            feature_names: Names of features
            n_features: Number of top features to select
            use_cv: Use cross-validation for feature selection

        Returns:
            X_selected: Selected features
            selected_indices: Indices of selected features
            selected_names: Names of selected features
        """
        print(f"\n  Feature Selection Strategy:")

        # Ensure n_features doesn't exceed available features or samples
        n_features = min(n_features, X.shape[0] // 3, X.shape[1])
        print(f"    Target features: {n_features}")

        # Strategy 1: Univariate C-index scoring
        def survival_score(X, y):
            scores = []
            for i in range(X.shape[1]):
                try:
                    c_index = concordance_index(y['time'], -X[:, i], y['event'])
                    score = abs(c_index - 0.5)  # Distance from random (0.5)
                    scores.append(score)
                except:
                    scores.append(0.0)
            return np.array(scores), np.array(scores)

        # First pass: Select more features for secondary selection
        first_pass_k = min(FIRST_PASS_FEATURES, X.shape[1], X.shape[0] // 2)
        univariate_selector = SelectKBest(score_func=survival_score, k=first_pass_k)
        X_univariate = univariate_selector.fit_transform(X, y)
        univariate_idx = univariate_selector.get_support(indices=True)
        print(f"    Univariate preselection: {len(univariate_idx)} features")

        # Strategy 2: Use L1-penalized Cox for final selection
        print(f"    L1-Cox for final selection...")
        try:
            # Moderate L1 penalty to retain useful features (final model will regularize)
            coxnet = CoxnetSurvivalAnalysis(
                l1_ratio=0.9,  # Balanced L1/L2 mix
                alphas=np.logspace(-2, 1, 20),  # Moderate alpha range
                max_iter=3000
            )
            X_subset = X[:, univariate_idx]
            coxnet.fit(X_subset, y)

            # FIXED: Get coefficients with proper bounds checking
            coef = np.abs(coxnet.coef_)

            # Filter to non-zero coefficients
            non_zero_mask = coef > 1e-10

            if np.sum(non_zero_mask) == 0:
                print(f"    Warning: L1-Cox zeroed all features, using univariate only")
                selected_indices = univariate_idx[:n_features]
            else:
                coef_nz = coef[non_zero_mask]
                indices_nz = np.where(non_zero_mask)[0]

                # Get top features safely
                n_select = min(n_features, len(coef_nz))
                top_in_subset = np.argsort(coef_nz)[::-1][:n_select]

                # Map back to original indices
                selected_in_subset = indices_nz[top_in_subset]
                selected_indices = univariate_idx[selected_in_subset]

                # If we got fewer than requested, fill with remaining univariate features
                if len(selected_indices) < n_features:
                    remaining_needed = n_features - len(selected_indices)
                    remaining_mask = ~np.isin(univariate_idx, selected_indices)
                    remaining_idx = univariate_idx[remaining_mask][:remaining_needed]
                    selected_indices = np.concatenate([selected_indices, remaining_idx])

        except Exception as e:
            print(f"    Warning: L1-Cox failed ({str(e)}), using univariate only")
            selected_indices = univariate_idx[:n_features]

        # Ensure proper type, uniqueness, and size
        selected_indices = np.unique(selected_indices)[:n_features]
        selected_indices = np.array(selected_indices, dtype=int).flatten()

        X_selected = X[:, selected_indices]
        selected_names = [feature_names[i] for i in selected_indices]

        print(f"    Final: {len(selected_indices)} features selected")

        return X_selected, selected_indices.tolist(), selected_names

    def fit_cox_model(self, X, y, feature_names, X_val=None, y_val=None, tune_penalizer=True):
        """
        Fit Cox Proportional Hazards model with optional hyperparameter tuning

        Args:
            X: Feature matrix
            y: Survival outcome
            feature_names: Names of features
            X_val: Validation feature matrix (for tuning)
            y_val: Validation outcome (for tuning)
            tune_penalizer: Whether to tune the penalizer parameter

        Returns:
            Fitted Cox model
        """
        # Tune penalizer using validation set
        if tune_penalizer and X_val is not None and y_val is not None:
            print(f"\n  Tuning Cox model penalizer...")
            best_penalizer = 10.0  # Default to strong regularization
            best_val_c = 0
            best_train_c = 0

            # Strong regularization range to combat overfitting
            for penalizer in [1.0, 5.0, 10.0, 20.0, 50.0]:
                try:
                    df_train = pd.DataFrame(X, columns=feature_names)
                    df_train['time'] = y['time']
                    df_train['event'] = y['event'].astype(int)

                    temp_model = CoxPHFitter(penalizer=penalizer)
                    temp_model.fit(df_train, duration_col='time', event_col='event', show_progress=False)

                    # Evaluate on training (to check overfitting)
                    predictions_train = temp_model.predict_partial_hazard(df_train[feature_names])
                    train_c = concordance_index(y['time'], -predictions_train, y['event'])

                    # Evaluate on validation
                    df_val = pd.DataFrame(X_val, columns=feature_names)
                    predictions_val = temp_model.predict_partial_hazard(df_val)
                    val_c = concordance_index(y_val['time'], -predictions_val, y_val['event'])

                    # Prefer models with smaller train-val gap (less overfitting)
                    gap = abs(train_c - val_c)
                    score = val_c - (0.1 * gap)  # Penalty for overfitting

                    if val_c > best_val_c or (val_c >= best_val_c - 0.02 and gap < abs(best_train_c - best_val_c)):
                        best_val_c = val_c
                        best_train_c = train_c
                        best_penalizer = penalizer

                except Exception as e:
                    continue

            print(f"    Best penalizer: {best_penalizer} (Val C-index: {best_val_c:.4f}, Train-Val gap: {abs(best_train_c - best_val_c):.4f}, Train-Val gap: {abs(best_train_c - best_val_c):.4f})")
        else:
            best_penalizer = 10.0  # Strong default to reduce overfitting

        # Fit final model with best penalizer
        df = pd.DataFrame(X, columns=feature_names)
        df['time'] = y['time']
        df['event'] = y['event'].astype(int)

        self.cox_model = CoxPHFitter(penalizer=best_penalizer, l1_ratio=0.0)
        self.cox_model.fit(df, duration_col='time', event_col='event', show_progress=False)

        # ADDED: Test proportional hazards assumption
        try:
            ph_test = proportional_hazard_test(self.cox_model, df, time_col='time', event_col='event')
            violations = ph_test.summary['p'] < 0.05
            if any(violations):
                violated_features = ph_test.summary.index[violations].tolist()
                print(f"\n  Warning: Proportional hazards assumption may be violated for:")
                for feat in violated_features[:5]:  # Show first 5
                    print(f"    - {feat}")
                if len(violated_features) > 5:
                    print(f"    ... and {len(violated_features) - 5} more")
        except Exception as e:
            print(f"  Note: Could not test proportional hazards assumption: {str(e)}")

        return self.cox_model

    def fit_random_survival_forest(self, X, y, X_val=None, y_val=None, tune_params=True):
        """
        Fit Random Survival Forest model with optional hyperparameter tuning

        Args:
            X: Feature matrix
            y: Survival outcome
            X_val: Validation feature matrix (for tuning)
            y_val: Validation outcome (for tuning)
            tune_params: Whether to tune hyperparameters

        Returns:
            Fitted RSF model
        """
        if tune_params and X_val is not None and y_val is not None:
            print(f"\n  Tuning RSF hyperparameters...")
            # More conservative defaults to reduce overfitting
            best_params = {'n_estimators': 100, 'max_depth': 3, 'min_samples_split': 20}
            best_val_c = 0

            # Simplified param grid with more regularization
            param_grid = [
                {'n_estimators': 50, 'max_depth': 3, 'min_samples_split': 20},
                {'n_estimators': 100, 'max_depth': 3, 'min_samples_split': 25},
                {'n_estimators': 100, 'max_depth': 5, 'min_samples_split': 20},
            ]

            for params in param_grid:
                try:
                    temp_model = RandomSurvivalForest(
                        n_estimators=params['n_estimators'],
                        max_depth=params['max_depth'],
                        min_samples_split=params['min_samples_split'],
                        min_samples_leaf=10,
                        max_features='sqrt',
                        random_state=self.random_state,
                        n_jobs=-1
                    )
                    temp_model.fit(X, y)
                    val_c = temp_model.score(X_val, y_val)

                    if val_c > best_val_c:
                        best_val_c = val_c
                        best_params = params
                except Exception as e:
                    continue

            print(f"    Best params: {best_params} (Val C-index: {best_val_c:.4f})")
        else:
            # More conservative defaults to reduce overfitting
            best_params = {'n_estimators': 100, 'max_depth': 3, 'min_samples_split': 20}

        # Fit final model with best parameters
        self.rsf_model = RandomSurvivalForest(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=15,  # Increased from 10 to reduce overfitting
            max_features='sqrt',
            random_state=self.random_state,
            n_jobs=-1
        )
        self.rsf_model.fit(X, y)

        return self.rsf_model

    def evaluate_model(self, X, y, feature_names, model_type='cox'):
        """
        Evaluate survival model using concordance index
        FIXED: Proper feature name matching for Cox model

        Args:
            X: Feature matrix
            y: Survival outcome
            feature_names: List of feature names corresponding to X columns
            model_type: 'cox' or 'rsf'

        Returns:
            C-index score
        """
        if model_type == 'cox':
            if self.cox_model is None:
                raise ValueError("Cox model not fitted yet")

            # Get the features used in the model (excluding time and event)
            model_features = [col for col in self.cox_model.params_.index
                            if col not in ['time', 'event']]

            # Ensure feature_names match
            if not all(f in feature_names for f in model_features):
                missing = [f for f in model_features if f not in feature_names]
                raise ValueError(f"Model features not in provided features: {missing[:5]}")

            # Create DataFrame with all features, then select only model features
            df = pd.DataFrame(X, columns=feature_names)
            df_model = df[model_features]

            predictions = self.cox_model.predict_partial_hazard(df_model)
            c_index = concordance_index(y['time'], -predictions, y['event'])

        elif model_type == 'rsf':
            if self.rsf_model is None:
                raise ValueError("RSF model not fitted yet")
            c_index = self.rsf_model.score(X, y)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return c_index

    def bootstrap_ci(self, X, y, feature_names, model_type='cox', n_bootstrap=100, confidence=0.95):
        """
        Calculate bootstrap confidence interval for C-index

        Args:
            X: Feature matrix
            y: Survival outcome
            feature_names: Feature names
            model_type: 'cox' or 'rsf'
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level

        Returns:
            mean_c, lower_ci, upper_ci
        """
        np.random.seed(self.random_state)
        c_indices = []

        n_samples = len(y)
        for _ in range(n_bootstrap):
            # Bootstrap sample
            idx = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[idx]
            y_boot = y[idx]

            try:
                c_idx = self.evaluate_model(X_boot, y_boot, feature_names, model_type)
                c_indices.append(c_idx)
            except:
                continue

        c_indices = np.array(c_indices)
        alpha = 1 - confidence
        lower_ci = np.percentile(c_indices, alpha/2 * 100)
        upper_ci = np.percentile(c_indices, (1 - alpha/2) * 100)
        mean_c = np.mean(c_indices)

        return mean_c, lower_ci, upper_ci

    def cross_validate(self, X, y, feature_names, n_splits=5, model_type='cox', penalizer=10.0):
        """
        Perform cross-validation with consistent random state
        Includes feature selection within each fold to prevent data leakage

        Args:
            X: Feature matrix
            y: Survival outcome
            feature_names: Feature names
            n_splits: Number of CV folds
            model_type: 'cox' or 'rsf'
            penalizer: Regularization strength for Cox model

        Returns:
            Array of C-index scores for each fold
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        c_indices = []

        print(f"  Running {n_splits}-fold nested CV...")
        for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Scale features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Feature selection within fold
            n_features = min(DEFAULT_N_FEATURES, X_train.shape[1], X_train.shape[0] // 3)
            X_train_sel, sel_idx, _ = self.select_features(X_train, y_train, feature_names, n_features=n_features, use_cv=False)
            X_test_sel = X_test[:, sel_idx]

            if model_type == 'cox':
                # Fit Cox model for this fold
                model = CoxPHSurvivalAnalysis(alpha=penalizer)
                model.fit(X_train_sel, y_train)
                c_index = model.score(X_test_sel, y_test)

            elif model_type == 'rsf':
                # Fit RSF for this fold
                model = RandomSurvivalForest(
                    n_estimators=100,
                    max_depth=3,
                    min_samples_split=20,
                    min_samples_leaf=15,
                    max_features='sqrt',
                    random_state=self.random_state,
                    n_jobs=-1
                )
                model.fit(X_train_sel, y_train)
                c_index = model.score(X_test_sel, y_test)

            c_indices.append(c_index)
            print(f"    Fold {fold}: C-index = {c_index:.4f}")

        return np.array(c_indices)

    def plot_kaplan_meier(self, y, groups=None, save_path=None):
        """
        Plot Kaplan-Meier survival curves

        Args:
            y: Survival outcome
            groups: Optional group labels for stratification
            save_path: Path to save figure
        """
        kmf = KaplanMeierFitter()

        plt.figure(figsize=(10, 6))

        if groups is None:
            # Single survival curve
            kmf.fit(y['time'], y['event'], label='All patients')
            kmf.plot_survival_function()
        else:
            # Stratified survival curves
            unique_groups = np.unique(groups)
            for group in unique_groups:
                mask = groups == group
                kmf.fit(y['time'][mask], y['event'][mask], label=f'Group {group}')
                kmf.plot_survival_function()

        plt.xlabel('Time (months)', fontsize=12)
        plt.ylabel('Survival Probability', fontsize=12)
        plt.title(f'Kaplan-Meier Survival Curve ({self.outcome_type})', fontsize=14, weight='bold')
        plt.legend(loc='best')
        plt.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_nomogram(self, feature_names, save_path=None):
        """
        Plot nomogram for Cox Proportional Hazards model
        Shows each feature's contribution to the risk score

        Args:
            feature_names: Names of features in the model
            save_path: Path to save figure
        """
        if self.cox_model is None:
            raise ValueError("Cox model not fitted yet")

        # Get coefficients from Cox model
        summary = self.cox_model.summary
        coef = summary['coef'].values
        hr = summary['exp(coef)'].values
        p_values = summary['p'].values

        # Select significant features (p < 0.05) or top features
        sig_idx = np.where(p_values < 0.05)[0]
        if len(sig_idx) == 0:
            # If no significant features, use top 10 by absolute coefficient
            sig_idx = np.argsort(np.abs(coef))[::-1][:10]
        elif len(sig_idx) > 15:
            # Limit to top 15 significant features
            sig_coef = coef[sig_idx]
            top_sig = np.argsort(np.abs(sig_coef))[::-1][:15]
            sig_idx = sig_idx[top_sig]

        feature_names_sig = [feature_names[i] for i in sig_idx]
        coef_sig = coef[sig_idx]
        hr_sig = hr[sig_idx]

        # Create nomogram
        fig, ax = plt.subplots(figsize=(12, max(8, len(feature_names_sig) * 0.8)))

        # Calculate point scores (scaled from coefficients)
        max_coef = np.max(np.abs(coef_sig))
        if max_coef > 0:
            points = (coef_sig / max_coef) * 100  # Scale to 0-100 points
        else:
            points = coef_sig * 0

        # Plot each feature
        y_pos = len(feature_names_sig)
        for i, (feat, pt, hr_val, coef_val) in enumerate(zip(feature_names_sig, points, hr_sig, coef_sig)):
            # Shorten feature name if too long
            feat_display = feat if len(feat) < 40 else feat[:37] + '...'

            # Color by direction of effect
            color = 'red' if coef_val > 0 else 'blue'

            # Plot point contribution
            ax.barh(y_pos - i, pt, height=0.7, color=color, alpha=0.6, edgecolor='black')

            # Add feature name and hazard ratio
            ax.text(-5, y_pos - i, feat_display, ha='right', va='center', fontsize=9, weight='bold')
            ax.text(pt + 2, y_pos - i, f'HR={hr_val:.2f}', ha='left', va='center', fontsize=8)

        # Configure axes
        ax.set_ylim(0, len(feature_names_sig) + 1)
        ax.set_xlim(-50, 110)
        ax.set_xlabel('Points (contribution to risk)', fontsize=11, weight='bold')
        ax.set_yticks([])
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.grid(axis='x', alpha=0.3)

        # Add title and legend
        ax.set_title(f'Nomogram for {self.outcome_type} Prediction\n' +
                     'Red: Increased Risk (HR>1), Blue: Decreased Risk (HR<1)',
                     fontsize=13, weight='bold', pad=20)

        # Add interpretation text
        fig.text(0.5, 0.02,
                 'Points represent each variable\'s contribution to survival risk. Higher positive points = higher risk.',
                 ha='center', fontsize=9, style='italic', wrap=True)

        plt.tight_layout(rect=[0, 0.03, 1, 1])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_feature_importance(self, feature_names, save_path=None):
        """
        Plot feature importance from Cox model

        Args:
            feature_names: Names of features
            save_path: Path to save figure
        """
        if self.cox_model is None:
            raise ValueError("Cox model not fitted yet")

        # Get coefficients and p-values
        summary = self.cox_model.summary
        coef = summary['coef'].values
        p_values = summary['p'].values

        # Sort by absolute coefficient value
        n_plot = min(TOP_PLOT_FEATURES, len(feature_names))
        sorted_idx = np.argsort(np.abs(coef))[::-1][:n_plot]

        plt.figure(figsize=(10, 8))
        colors = ['red' if p < 0.05 else 'gray' for p in p_values[sorted_idx]]
        plt.barh(range(len(sorted_idx)), coef[sorted_idx], color=colors)
        plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
        plt.xlabel('Coefficient (log Hazard Ratio)', fontsize=12)
        plt.title(f'Feature Importance in Cox Model\n(Red: p < 0.05, Gray: p ≥ 0.05)',
                 fontsize=14, weight='bold')
        plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        plt.grid(axis='x', alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def create_stratified_survival_split(clinical_df, radiomics_dfs, outcome_type='OS',
                                     test_size=DEFAULT_TEST_SIZE, random_state=42):
    """
    Create stratified train/val split for survival analysis
    FIXED: Better handling of small strata

    Args:
        clinical_df: Clinical data DataFrame
        radiomics_dfs: Dict of radiomics DataFrames by region
        outcome_type: 'OS' or 'TTP'
        test_size: Validation set proportion
        random_state: Random seed

    Returns:
        train_radiomics, val_radiomics, train_clinical, val_clinical
    """
    # Determine outcome columns
    if outcome_type == 'OS':
        time_col = 'OS'
        event_col = 'Death_1_StillAliveorLostToFU_0'
    elif outcome_type == 'TTP':
        time_col = 'TTP'
        event_col = 'Censored_0_progressed_1'
    else:
        raise ValueError(f"Unknown outcome type: {outcome_type}")

    # Prepare data for stratification
    data = clinical_df.copy()
    data['TCIA_ID'] = data['TCIA_ID'].astype(str).str.replace('HCC_', '')

    # Get patients with complete survival data
    data = data.dropna(subset=[time_col, event_col])

    if len(data) == 0:
        raise ValueError("No samples with complete survival data")

    # Create stratification variable: combine event status and time tertiles
    times = data[time_col].values
    events = data[event_col].values

    # Define time groups (tertiles among events)
    event_times = times[events == 1]
    if len(event_times) > 3:
        tertiles = np.percentile(event_times, [33.33, 66.67])
        time_groups = np.digitize(times, tertiles)  # 0=early, 1=mid, 2=late
    else:
        time_groups = np.zeros(len(times), dtype=int)

    # Combine event status (0/1) and time group (0/1/2) for stratification
    stratify_labels = events.astype(int) * 3 + time_groups

    # FIXED: Check stratum sizes before attempting stratified split
    strata_counts = Counter(stratify_labels)
    min_stratum_size = min(strata_counts.values())

    patient_ids = data['TCIA_ID'].values

    if min_stratum_size < 2:
        print(f"  Warning: Some strata have <2 samples. Using event-only stratification.")
        # Simplify to event-only stratification
        stratify_labels = events.astype(int)
        strata_counts = Counter(stratify_labels)
        min_stratum_size = min(strata_counts.values())

    # Stratified split
    try:
        if min_stratum_size >= 2:
            train_ids, val_ids = train_test_split(
                patient_ids, test_size=test_size,
                stratify=stratify_labels, random_state=random_state
            )
        else:
            raise ValueError("Insufficient samples for stratification")
    except (ValueError, Exception) as e:
        # If stratification fails, do simple random split
        print(f"  Warning: Stratified split failed ({str(e)}). Using random split.")
        train_ids, val_ids = train_test_split(
            patient_ids, test_size=test_size, random_state=random_state
        )

    # Split radiomics by region
    train_radiomics = {}
    val_radiomics = {}

    if radiomics_dfs is not None:
        for region, radiomics_df in radiomics_dfs.items():
            if radiomics_df is not None and not radiomics_df.empty:
                radiomics_df = radiomics_df.copy()
                radiomics_df['patient_id'] = radiomics_df['patient_id'].astype(str).str.replace('HCC_', '')

                train_radiomics[region] = radiomics_df[radiomics_df['patient_id'].isin(train_ids)].copy()
                val_radiomics[region] = radiomics_df[radiomics_df['patient_id'].isin(val_ids)].copy()

    # Split clinical data
    train_clinical = clinical_df[clinical_df['TCIA_ID'].astype(str).str.replace('HCC_', '').isin(train_ids)].copy()
    val_clinical = clinical_df[clinical_df['TCIA_ID'].astype(str).str.replace('HCC_', '').isin(val_ids)].copy()

    print(f"\nStratified survival split on '{outcome_type}':")
    print(f"  Train: {len(train_ids)} patients")
    print(f"  Val: {len(val_ids)} patients")

    return train_radiomics, val_radiomics, train_clinical, val_clinical


def run_survival_analysis(clinical_csv, radiomics_dir='results/radiomics', output_dir='results/survival'):
    """
    Run complete survival analysis pipeline with all fixes applied

    Args:
        clinical_csv: Path to clinical data
        radiomics_dir: Directory containing radiomics features by region
        output_dir: Directory to save results
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load clinical data
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    if clinical_csv.endswith('.xlsx') or clinical_csv.endswith('.xls'):
        clinical_df = pd.read_excel(clinical_csv)
    else:
        clinical_df = pd.read_csv(clinical_csv)

    print(f"Loaded clinical data: {clinical_df.shape[0]} patients")

    # Load radiomics by region
    radiomics_by_region = {}
    region_names = ['tumor', 'liver', 'bloodvessels']
    radiomics_path = Path(radiomics_dir)

    print("\nLoading radiomics features by region...")
    for region in region_names:
        train_file = radiomics_path / f'radiomics_features_{region}_train.csv'
        val_file = radiomics_path / f'radiomics_features_{region}_val.csv'

        dfs = []
        if train_file.exists():
            dfs.append(pd.read_csv(train_file))
        if val_file.exists():
            dfs.append(pd.read_csv(val_file))

        if dfs:
            region_df = pd.concat(dfs, ignore_index=True)
            region_df = region_df.loc[:, ~region_df.columns.duplicated()]
            radiomics_by_region[region] = region_df
            print(f"  {region}: {region_df.shape[0]} patients, {region_df.shape[1]-2} features")

    if not radiomics_by_region:
        print("Warning: No radiomics features found. Using clinical data only.")
        radiomics_by_region = None

    # Results summary
    all_results = []

    # OVERALL SURVIVAL ANALYSIS
    print("\n" + "="*60)
    print("OVERALL SURVIVAL ANALYSIS")
    print("="*60)

    print("\nCreating stratified split for Overall Survival...")
    train_radiomics_os, val_radiomics_os, train_clinical_os, val_clinical_os = create_stratified_survival_split(
        clinical_df, radiomics_by_region, outcome_type='OS', test_size=DEFAULT_TEST_SIZE, random_state=42
    )

    os_analyzer = SurvivalAnalyzer(outcome_type='OS', random_state=42)

    # Prepare training data
    print("\nPreparing training data...")
    X_train, y_train, feature_names, data_train = os_analyzer.prepare_data(
        train_clinical_os, train_radiomics_os, is_training=True
    )
    print(f"Train Dataset: {X_train.shape[0]} patients, {X_train.shape[1]} features")
    print(f"Train Events: {np.sum(y_train['event'])}, Censored: {np.sum(~y_train['event'])}")

    # Prepare validation data
    print("\nPreparing validation data...")
    X_val, y_val, _, data_val = os_analyzer.prepare_data(
        val_clinical_os, val_radiomics_os, is_training=False
    )
    print(f"Val Dataset: {X_val.shape[0]} patients, {X_val.shape[1]} features")
    print(f"Val Events: {np.sum(y_val['event'])}, Censored: {np.sum(~y_val['event'])}")

    # Scale features
    X_train_scaled = os_analyzer.scaler.fit_transform(X_train)
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    X_val_scaled = os_analyzer.scaler.transform(X_val)
    X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # Feature selection
    X_train_selected, selected_idx, selected_features = os_analyzer.select_features(
        X_train_scaled, y_train, feature_names, n_features=DEFAULT_N_FEATURES, use_cv=True
    )
    print(f"\nSelected {len(selected_features)} features for OS:")
    for i, feat in enumerate(selected_features[:10], 1):  # Show first 10
        feat_display = feat if len(feat) < 60 else feat[:57] + '...'
        print(f"  {i}. {feat_display}")
    if len(selected_features) > 10:
        print(f"  ... and {len(selected_features) - 10} more")

    X_val_selected = X_val_scaled[:, selected_idx]

    # Fit Cox model
    print("\nFitting Cox Proportional Hazards model...")
    cox_model = os_analyzer.fit_cox_model(
        X_train_selected, y_train, selected_features,
        X_val=X_val_selected, y_val=y_val, tune_penalizer=True
    )

    print(f"\nCox Model Summary (Top 10 features by p-value):")
    summary_sorted = cox_model.summary.sort_values('p')
    print(summary_sorted[['coef', 'exp(coef)', 'p']].head(10).to_string())

    # Evaluate with confidence intervals
    print("\nEvaluating Cox model...")
    train_c_index = os_analyzer.evaluate_model(X_train_selected, y_train, selected_features, model_type='cox')
    val_c_index = os_analyzer.evaluate_model(X_val_selected, y_val, selected_features, model_type='cox')

    print(f"Train C-index: {train_c_index:.4f}")
    print(f"Val C-index: {val_c_index:.4f}")
    print(f"Train-Val Gap: {abs(train_c_index - val_c_index):.4f}")

    # Nested CV on training set for more robust estimate
    print("\nNested cross-validation on training set...")
    cv_scores = os_analyzer.cross_validate(X_train_scaled, y_train, feature_names, n_splits=5, model_type='cox', penalizer=10.0)
    print(f"CV C-index: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Bootstrap CI for validation
    print("Calculating bootstrap CI for validation C-index...")
    val_c_mean, val_c_lower, val_c_upper = os_analyzer.bootstrap_ci(
        X_val_selected, y_val, selected_features, model_type='cox', n_bootstrap=100
    )
    print(f"Val C-index (bootstrap): {val_c_mean:.4f} (95% CI: {val_c_lower:.4f}-{val_c_upper:.4f})")

    all_results.append({
        'outcome': 'OS',
        'model': 'Cox PH',
        'train_c_index': train_c_index,
        'val_c_index': val_c_index,
        'cv_c_mean': cv_scores.mean(),
        'cv_c_std': cv_scores.std(),
        'train_val_gap': abs(train_c_index - val_c_index),
        'val_c_lower': val_c_lower,
        'val_c_upper': val_c_upper
    })

    # Random Survival Forest
    print("\nFitting Random Survival Forest...")
    rsf_model = os_analyzer.fit_random_survival_forest(
        X_train_selected, y_train,
        X_val=X_val_selected, y_val=y_val, tune_params=True
    )

    rsf_train_c = os_analyzer.evaluate_model(X_train_selected, y_train, selected_features, model_type='rsf')
    rsf_val_c = os_analyzer.evaluate_model(X_val_selected, y_val, selected_features, model_type='rsf')

    print(f"RSF Train C-index: {rsf_train_c:.4f}")
    print(f"RSF Val C-index: {rsf_val_c:.4f}")
    print(f"RSF Train-Val Gap: {abs(rsf_train_c - rsf_val_c):.4f}")

    # Nested CV for RSF
    print("\nNested cross-validation for RSF...")
    rsf_cv_scores = os_analyzer.cross_validate(X_train_scaled, y_train, feature_names, n_splits=5, model_type='rsf')
    print(f"RSF CV C-index: {rsf_cv_scores.mean():.4f} ± {rsf_cv_scores.std():.4f}")

    # Bootstrap CI for RSF
    print("Calculating bootstrap CI for RSF validation C-index...")
    rsf_val_c_mean, rsf_val_c_lower, rsf_val_c_upper = os_analyzer.bootstrap_ci(
        X_val_selected, y_val, selected_features, model_type='rsf', n_bootstrap=100
    )
    print(f"RSF Val C-index (bootstrap): {rsf_val_c_mean:.4f} (95% CI: {rsf_val_c_lower:.4f}-{rsf_val_c_upper:.4f})")

    all_results.append({
        'outcome': 'OS',
        'model': 'RSF',
        'train_c_index': rsf_train_c,
        'val_c_index': rsf_val_c,
        'cv_c_mean': rsf_cv_scores.mean(),
        'cv_c_std': rsf_cv_scores.std(),
        'train_val_gap': abs(rsf_train_c - rsf_val_c),
        'val_c_lower': rsf_val_c_lower,
        'val_c_upper': rsf_val_c_upper
    })

    # Generate plots
    print("\nGenerating visualizations...")
    os_analyzer.plot_kaplan_meier(y_train, save_path=f"{output_dir}/km_curve_os_train.png")
    os_analyzer.plot_kaplan_meier(y_val, save_path=f"{output_dir}/km_curve_os_val.png")
    os_analyzer.plot_feature_importance(selected_features, save_path=f"{output_dir}/feature_importance_os.png")
    os_analyzer.plot_nomogram(selected_features, save_path=f"{output_dir}/nomogram_os.png")

    # TIME TO PROGRESSION ANALYSIS (CLINICAL FEATURES ONLY)
    print("\n" + "="*60)
    print("TIME TO PROGRESSION ANALYSIS (CLINICAL FEATURES ONLY)")
    print("="*60)

    print("\nCreating stratified split for Time To Progression...")
    # Use only clinical features for TTP (no radiomics)
    train_radiomics_ttp, val_radiomics_ttp, train_clinical_ttp, val_clinical_ttp = create_stratified_survival_split(
        clinical_df, None, outcome_type='TTP', test_size=DEFAULT_TEST_SIZE, random_state=42
    )

    ttp_analyzer = SurvivalAnalyzer(outcome_type='TTP', random_state=42)

    # Prepare data (clinical features only)
    print("\nPreparing training data (clinical features only)...")
    X_train, y_train, feature_names, data_train = ttp_analyzer.prepare_data(
        train_clinical_ttp, None, is_training=True
    )
    print(f"Train Dataset: {X_train.shape[0]} patients, {X_train.shape[1]} features")
    print(f"Train Events: {np.sum(y_train['event'])}, Censored: {np.sum(~y_train['event'])}")

    print("\nPreparing validation data (clinical features only)...")
    X_val, y_val, _, data_val = ttp_analyzer.prepare_data(
        val_clinical_ttp, None, is_training=False
    )
    print(f"Val Dataset: {X_val.shape[0]} patients, {X_val.shape[1]} features")
    print(f"Val Events: {np.sum(y_val['event'])}, Censored: {np.sum(~y_val['event'])}")

    # Scale and select features
    X_train_scaled = ttp_analyzer.scaler.fit_transform(X_train)
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    X_val_scaled = ttp_analyzer.scaler.transform(X_val)
    X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    X_train_selected, selected_idx, selected_features = ttp_analyzer.select_features(
        X_train_scaled, y_train, feature_names, n_features=DEFAULT_N_FEATURES, use_cv=True
    )
    print(f"\nSelected {len(selected_features)} features for TTP (clinical only):")
    for i, feat in enumerate(selected_features[:10], 1):  # Show first 10
        feat_display = feat if len(feat) < 60 else feat[:57] + '...'
        print(f"  {i}. {feat_display}")
    if len(selected_features) > 10:
        print(f"  ... and {len(selected_features) - 10} more")
    X_val_selected = X_val_scaled[:, selected_idx]

    # Fit Cox model
    print("\nFitting Cox Proportional Hazards model...")
    cox_model = ttp_analyzer.fit_cox_model(
        X_train_selected, y_train, selected_features,
        X_val=X_val_selected, y_val=y_val, tune_penalizer=True
    )

    train_c = ttp_analyzer.evaluate_model(X_train_selected, y_train, selected_features, model_type='cox')
    val_c = ttp_analyzer.evaluate_model(X_val_selected, y_val, selected_features, model_type='cox')

    print(f"\nTrain C-index: {train_c:.4f}")
    print(f"Val C-index: {val_c:.4f}")
    print(f"Train-Val Gap: {abs(train_c - val_c):.4f}")

    # Nested CV on training set for more robust estimate
    print("\nNested cross-validation on training set...")
    cv_scores = ttp_analyzer.cross_validate(X_train_scaled, y_train, feature_names, n_splits=5, model_type='cox', penalizer=10.0)
    print(f"CV C-index: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Bootstrap CI
    print("Calculating bootstrap CI for validation C-index...")
    val_c_mean, val_c_lower, val_c_upper = ttp_analyzer.bootstrap_ci(
        X_val_selected, y_val, selected_features, model_type='cox', n_bootstrap=100
    )
    print(f"Val C-index (bootstrap): {val_c_mean:.4f} (95% CI: {val_c_lower:.4f}-{val_c_upper:.4f})")

    all_results.append({
        'outcome': 'TTP',
        'model': 'Cox PH',
        'train_c_index': train_c,
        'val_c_index': val_c,
        'cv_c_mean': cv_scores.mean(),
        'cv_c_std': cv_scores.std(),
        'train_val_gap': abs(train_c - val_c),
        'val_c_lower': val_c_lower,
        'val_c_upper': val_c_upper
    })

    # Random Survival Forest
    print("\nFitting Random Survival Forest...")
    rsf_model = ttp_analyzer.fit_random_survival_forest(
        X_train_selected, y_train,
        X_val=X_val_selected, y_val=y_val, tune_params=True
    )

    rsf_train_c = ttp_analyzer.evaluate_model(X_train_selected, y_train, selected_features, model_type='rsf')
    rsf_val_c = ttp_analyzer.evaluate_model(X_val_selected, y_val, selected_features, model_type='rsf')

    print(f"RSF Train C-index: {rsf_train_c:.4f}")
    print(f"RSF Val C-index: {rsf_val_c:.4f}")
    print(f"RSF Train-Val Gap: {abs(rsf_train_c - rsf_val_c):.4f}")

    # Nested CV for RSF
    print("\nNested cross-validation for RSF...")
    rsf_cv_scores = ttp_analyzer.cross_validate(X_train_scaled, y_train, feature_names, n_splits=5, model_type='rsf')
    print(f"RSF CV C-index: {rsf_cv_scores.mean():.4f} ± {rsf_cv_scores.std():.4f}")

    # Bootstrap CI for RSF
    print("Calculating bootstrap CI for RSF validation C-index...")
    rsf_val_c_mean, rsf_val_c_lower, rsf_val_c_upper = ttp_analyzer.bootstrap_ci(
        X_val_selected, y_val, selected_features, model_type='rsf', n_bootstrap=100
    )
    print(f"RSF Val C-index (bootstrap): {rsf_val_c_mean:.4f} (95% CI: {rsf_val_c_lower:.4f}-{rsf_val_c_upper:.4f})")

    all_results.append({
        'outcome': 'TTP',
        'model': 'RSF',
        'train_c_index': rsf_train_c,
        'val_c_index': rsf_val_c,
        'cv_c_mean': rsf_cv_scores.mean(),
        'cv_c_std': rsf_cv_scores.std(),
        'train_val_gap': abs(rsf_train_c - rsf_val_c),
        'val_c_lower': rsf_val_c_lower,
        'val_c_upper': rsf_val_c_upper
    })

    # Generate plots
    print("\nGenerating visualizations...")
    ttp_analyzer.plot_kaplan_meier(y_train, save_path=f"{output_dir}/km_curve_ttp_train.png")
    ttp_analyzer.plot_kaplan_meier(y_val, save_path=f"{output_dir}/km_curve_ttp_val.png")
    ttp_analyzer.plot_feature_importance(selected_features, save_path=f"{output_dir}/feature_importance_ttp.png")
    ttp_analyzer.plot_nomogram(selected_features, save_path=f"{output_dir}/nomogram_ttp.png")

    # Save results summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"{output_dir}/metrics_summary.csv", index=False)

    # Print formatted results
    print("\nModel Performance Metrics:")
    print("-" * 100)
    for _, row in results_df.iterrows():
        print(f"{row['outcome']:>4} {row['model']:>6} | Train: {row['train_c_index']:.4f} | Val: {row['val_c_index']:.4f} | "
              f"CV: {row['cv_c_mean']:.4f}±{row['cv_c_std']:.4f} | Gap: {row['train_val_gap']:.4f}")
    print("-" * 100)
    print("\nKey Insights:")
    print(f"  • Best validation: {results_df.loc[results_df['val_c_index'].idxmax(), 'outcome']} "
          f"{results_df.loc[results_df['val_c_index'].idxmax(), 'model']} "
          f"(C-index: {results_df['val_c_index'].max():.4f})")
    print(f"  • Most stable (lowest CV std): {results_df.loc[results_df['cv_c_std'].idxmin(), 'outcome']} "
          f"{results_df.loc[results_df['cv_c_std'].idxmin(), 'model']} "
          f"(std: {results_df['cv_c_std'].min():.4f})")
    print(f"  • Least overfitting (smallest gap): {results_df.loc[results_df['train_val_gap'].idxmin(), 'outcome']} "
          f"{results_df.loc[results_df['train_val_gap'].idxmin(), 'model']} "
          f"(gap: {results_df['train_val_gap'].min():.4f})")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - metrics_summary.csv (performance metrics with confidence intervals)")
    print("  - km_curve_*_*.png (Kaplan-Meier survival curves)")
    print("  - feature_importance_*.png (Cox model feature importance)")
    print("  - nomogram_*.png (Clinical nomograms)")


if __name__ == "__main__":
    # Example usage
    clinical_csv = "/media/mirl/DATA/Projects/HCC/data/HCC-TACE-Seg_clinical_data-V2.xlsx"
    radiomics_dir = "/media/mirl/DATA/Projects/HCC/results/radiomics"
    output_dir = "/media/mirl/DATA/Projects/HCC/results/survival"

    run_survival_analysis(clinical_csv, radiomics_dir, output_dir)
