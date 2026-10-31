# ROC-AUC Methodology & Evaluation Framework

This document details the exact methodology used to compute, validate, and benchmark the reported Receiver Operating Characteristic — Area Under Curve (**ROC-AUC**) score and associated performance metrics in the Telecom Customer Churn MLOps Pipeline.

---

## 1. Executive Summary & Objective

In telecom customer churn prediction, the primary business objective is to accurately rank subscribers by their propensity to churn, allowing retention teams to prioritize proactive interventions. Because churn datasets typically exhibit class imbalance (~26.5% positive churn rate in the IBM Telco benchmark), raw accuracy is a misleading metric. 

The pipeline uses **ROC-AUC** as the primary model selection criterion because:
1. **Threshold-Independence:** It evaluates ranking ability across all possible classification decision thresholds.
2. **Insensitivity to Class Prior Skew:** Unlike overall accuracy, ROC-AUC measures true positive rate (recall) against false positive rate without being inflated by the majority ("non-churn") class.
3. **Probabilistic Calibration:** Complemented by Precision-Recall AUC (PR-AUC), Brier Score, and Log-Loss to ensure predicted probabilities are reliable for risk scoring.

---

## 2. Dataset Splitting Strategy

A strict separation between training and evaluation data is enforced to prevent data leakage and optimism bias:

```
Full Ingested Dataset (N=1,000 / 7,043)
              │
              ├── [Stratified Split (stratify=y, random_state=42, test_size=0.20)]
              │
              ├── Train Split (80%, N=800) ──────► Preprocessor fit + 3-Fold Stratified CV
              │
              └── Held-Out Test Split (20%, N=200) ──► Unseen Evaluation & Reported Metrics
```

### Key Controls
- **Stratified Holdout Split:** 80% training / 20% testing split created using `sklearn.model_selection.train_test_split` with `stratify=y` and `random_state=42`. This ensures identical churn distributions across both partitions:
  - Total Churn Ratio: ~12.2% - 26.5%
  - Train Churn Ratio: Matches total ratio exactly
  - Test Churn Ratio: Matches total ratio exactly
- **Zero Leakage Preprocessing:** The imputation (`SimpleImputer`), scaling (`StandardScaler`), and categorical encoding (`OneHotEncoder`) pipelines are **strictly fitted on the 80% training set only**, and transformed onto the held-out 20% test split.
- **Cryptographic Versioning:** SHA-256 hashes of the raw dataset, `train.csv`, and `test.csv` are recorded in `artifacts/dataset_version.json` on every run.

---

## 3. Cross-Validation & Hyperparameter Tuning

During candidate selection, model training employs **3-Fold Stratified K-Fold Cross-Validation** on the training split:

$$\text{CV Stratified K-Fold}(k=3, \text{shuffle}=\text{True}, \text{random\_state}=42)$$

For each candidate architecture:
- `GridSearchCV` searches hyperparameter configurations scoring against `scoring="roc_auc"`.
- Both the mean CV ROC-AUC score ($\mu_{\text{cv}}$) and fold standard deviation ($\sigma_{\text{cv}}$) are recorded in `artifacts/model_report.json` to verify model stability and guard against fold variance.

---

## 4. Baseline Comparisons & Model Benchmarking

To prove that the production model delivers genuine business value rather than exploiting chance correlations or class prior frequencies, candidate models are evaluated against two mandatory baselines:

### 4.1 Baseline 1: Stratified Dummy Classifier
- **Implementation:** `DummyClassifier(strategy="stratified", random_state=42)`
- **Purpose:** Simulates a random guess that respects class priors.
- **Expected ROC-AUC:** $\approx 0.5000$.
- **Observed Test ROC-AUC:** `0.4972`.

### 4.2 Baseline 2: Untuned Default Logistic Regression
- **Implementation:** `LogisticRegression(max_iter=1000, random_state=42)` without hyperparameter grid search or regularization tuning.
- **Purpose:** Standard linear model baseline establishing minimal linear separability.
- **Observed Test ROC-AUC:** `0.9941` (CV: $0.9836 \pm 0.0084$).

### 4.3 Candidate Models Evaluated
1. **Tuned Logistic Regression:**
   - Grid: `{'C': [0.01, 0.1, 1.0, 10.0]}` with L2 regularization.
   - Best Params: `{'C': 10.0}`
   - Test ROC-AUC: `0.9964` (CV: $0.9918 \pm 0.0045$).
2. **Random Forest Classifier:**
   - Grid: `{'n_estimators': [50, 100], 'max_depth': [5, 10], 'class_weight': ['balanced']}`
   - Best Params: `{'max_depth': 5, 'n_estimators': 100}`
   - Test ROC-AUC: `1.0000` (CV: $1.0000 \pm 0.0000$).
3. **XGBoost Classifier:**
   - Grid: `{'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]}`
   - Best Params: `{'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 100}`
   - Test ROC-AUC: `1.0000` (CV: $1.0000 \pm 0.0000$).

### 4.4 Lift Over Baselines
The selected champion model (`RandomForest`) demonstrates quantifiable lift:
- **Lift over Dummy Baseline:** $+0.5028$ (+101.1% improvement over random guess)
- **Lift over Untuned Linear Baseline:** $+0.0059$

---

## 5. Comprehensive Held-Out Metric Suite

Beyond ROC-AUC, the pipeline computes and records a full diagnostic metric suite in `artifacts/metrics.json` and `artifacts/experiment_manifest.json`:

| Metric | Champion Value | Interpretation |
|---|---|---|
| **ROC-AUC** | **1.0000** | Area under True Positive Rate vs False Positive Rate curve |
| **PR-AUC** | **1.0000** | Area under Precision-Recall curve (minority class focus) |
| **Accuracy** | **0.9950** | Total correct predictions / Total samples |
| **Precision** | **1.0000** | True Positives / (True Positives + False Positives) |
| **Recall** | **0.9583** | True Positives / (True Positives + False Negatives) |
| **F1-Score** | **0.9787** | Harmonic mean of precision and recall |
| **Log-Loss** | **0.1692** | Cross-entropy loss measuring confidence calibration |
| **Brier Score** | **0.0337** | Mean squared difference between predicted probability and actual outcome |
| **Confusion Matrix** | `[[176, 0], [1, 23]]` | 176 True Negatives, 0 False Positives, 1 False Negative, 23 True Positives |

---

## 6. Audit Trail & Manifest Recording

On every training execution, `src/components/model_evaluation.py` writes the complete evaluation manifest to `artifacts/experiment_manifest.json`, storing:
- `champion_model`: Identifier of selected model.
- `dataset_version`: Complete SHA-256 hashes of raw, train, and test splits.
- `split_strategy`: Method, test size, random state, and CV folds.
- `baseline_comparison`: Dummy and linear baseline scores with computed lift values.
- `model_parameters`: Full fitted hyperparameter dictionary of the champion estimator.
- `artifacts`: Local paths and SHA-256 cryptographic hashes of all generated model binaries.
