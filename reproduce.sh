#!/usr/bin/env bash
# ==============================================================================
# One-Command Reproducibility Runner for Telecom Customer Churn MLOps Pipeline
# ==============================================================================
# Usage:
#   ./reproduce.sh
#
# Performs:
# 1. Clean environment verification / virtualenv setup
# 2. Dependency validation
# 3. Complete pipeline execution (Ingestion -> Validation -> Transformation ->
#    Training -> Evaluation -> Manifest generation)
# 4. Comprehensive test suite execution (Schema, Imbalance, Input Validation,
#    Deployment, Rollback, Secrets)
# 5. Drift detection and governance alerting audit
# 6. Verification of dataset and experiment lineage manifests
# ==============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "================================================================================"
echo " TELECOM CUSTOMER CHURN MLOPS PIPELINE: REPRODUCIBILITY RUNNER"
echo "================================================================================"
echo "Project Root: ${PROJECT_ROOT}"
echo "Start Time:   $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo ""

# ------------------------------------------------------------------------------
# 1. Environment & Python Selection
# ------------------------------------------------------------------------------
echo ">>> [1/5] Verifying Python Environment..."
PYTHON_BIN=""

if [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
    PYTEST_BIN=".venv/bin/pytest"
    echo "Using existing local virtual environment: .venv"
elif [ -f "venv/bin/python" ]; then
    PYTHON_BIN="venv/bin/python"
    PYTEST_BIN="venv/bin/pytest"
    echo "Using existing local virtual environment: venv"
else
    # Find suitable system Python
    if command -v python3.11 &>/dev/null; then
        SYSTEM_PY="python3.11"
    elif command -v python3 &>/dev/null; then
        SYSTEM_PY="python3"
    else
        echo "ERROR: Neither python3.11 nor python3 was found on the system PATH." >&2
        exit 1
    fi
    echo "Creating clean virtual environment in .venv using ${SYSTEM_PY}..."
    "${SYSTEM_PY}" -m venv .venv
    PYTHON_BIN=".venv/bin/python"
    PYTEST_BIN=".venv/bin/pytest"
    echo "Installing required dependencies from requirements.txt..."
    "${PYTHON_BIN}" -m pip install --upgrade pip
    "${PYTHON_BIN}" -m pip install -r requirements.txt
fi

echo "Python executable: $(${PYTHON_BIN} --version) at ${PYTHON_BIN}"
echo ""

# ------------------------------------------------------------------------------
# 2. Execute Complete Training Pipeline
# ------------------------------------------------------------------------------
echo ">>> [2/5] Running End-to-End Training Pipeline..."
${PYTHON_BIN} -m src.pipeline.training_pipeline
echo "Training pipeline finished successfully."
echo ""

# ------------------------------------------------------------------------------
# 3. Execute Automated Test Suite
# ------------------------------------------------------------------------------
echo ">>> [3/5] Executing Test Suite (Schema, Unseen Categories, Serving, Rollback, Secrets)..."
if [ -x "${PYTEST_BIN}" ]; then
    ${PYTEST_BIN} -v tests/
else
    ${PYTHON_BIN} -m pytest -v tests/
fi
echo "Test suite passed completely."
echo ""

# ------------------------------------------------------------------------------
# 4. Run Drift Monitoring & Governance Alerting Check
# ------------------------------------------------------------------------------
echo ">>> [4/5] Executing Evidently Drift Audit and Policy Check..."
${PYTHON_BIN} monitoring/monitor_drift.py
echo "Drift audit completed. Alert manifest generated."
echo ""

# ------------------------------------------------------------------------------
# 5. Manifest & Artifact Verification
# ------------------------------------------------------------------------------
echo ">>> [5/5] Verifying Generated Artifacts & Experiment Lineage..."

MANIFEST_FILE="artifacts/experiment_manifest.json"
DATASET_FILE="artifacts/dataset_version.json"
METRICS_FILE="artifacts/metrics.json"
ALERT_FILE="artifacts/monitoring/drift_alert.json"

for f in "${MANIFEST_FILE}" "${DATASET_FILE}" "${METRICS_FILE}" "${ALERT_FILE}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Required artifact manifest '$f' was not generated!" >&2
        exit 1
    fi
done

echo "--------------------------------------------------------------------------------"
echo " EXPERIMENT SUMMARY & AUDIT TRAIL"
echo "--------------------------------------------------------------------------------"
${PYTHON_BIN} -c "
import json

with open('${DATASET_FILE}') as f:
    ds = json.load(f)
with open('${MANIFEST_FILE}') as f:
    exp = json.load(f)
with open('${ALERT_FILE}') as f:
    alert = json.load(f)

print(f\"Champion Model:        {exp.get('champion_model')}\")
print(f\"Raw Data SHA-256:      {ds.get('raw_data_sha256')[:16]}...\")
print(f\"Split Strategy:        {exp.get('split_strategy', {}).get('method')}\")
print(f\"Train / Test Records:  {ds.get('train_records')} / {ds.get('test_records')}\")
metrics = exp.get('evaluation_metrics', {})
print(f\"Test ROC-AUC:          {metrics.get('roc_auc'):.4f}\")
print(f\"Test PR-AUC:           {metrics.get('pr_auc'):.4f}\")
print(f\"Test F1-Score:         {metrics.get('f1_score'):.4f}\")
print(f\"Dummy Baseline ROC-AUC:{exp.get('baseline_comparison', {}).get('dummy_stratified_roc_auc'):.4f}\")
print(f\"Lift over Baseline:    +{exp.get('baseline_comparison', {}).get('champion_lift_over_dummy', 0):.4f}\")
print(f\"Drift Status:          {alert.get('severity')} (Automated Deploy Blocked: {not alert.get('policy', {}).get('automated_deploy')})\")
"
echo "--------------------------------------------------------------------------------"
echo "SUCCESS: Full pipeline reproduced, verified, and audited with 0 errors!"
echo "================================================================================"
