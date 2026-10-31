# Drift Response, Alerting, & Governance Review SOP

This document defines the monitoring architecture, alerting severity tiers, safety guardrails, and Standard Operating Procedure (SOP) for responding to distribution and prediction drift in the Telecom Customer Churn prediction service.

---

## 1. Safety Guardrail: Zero-Touch Retraining Ban

> [!CAUTION]
> **MANDATORY POLICY GUARDRAIL:**
> Under NO circumstances should the pipeline automatically retrain or deploy a new model simply because data or prediction drift is detected. Automated retraining without human review and label maturity verification is strictly prohibited.

### Why Automated Retraining on Drift is Dangerous:
1. **Label Maturity Window (Label Lag):** In subscription churn, ground-truth labels take **30 to 60 days** to mature (a subscriber must fail to renew or actively terminate service at the billing cycle boundary). Live production inference traffic has **no verified ground truth**. Auto-retraining on live traffic would require training on pseudo-labels or unverified outcomes, cementing model hallucinations.
2. **Upstream Schema and Telemetry Glitches:** A sudden distribution shift is often caused by upstream bugs (e.g. tracking SDK updates, currency conversion failures, front-end form validation errors, or missing feature columns). Automatically retraining on corrupted data permanently poisons the training distribution.
3. **Transient or Seasonal Anomalies:** Temporary external events (e.g. regional network outages, promotional marketing campaigns, Black Friday discount surges) cause temporary covariate shifts. Automatically retraining during a temporary anomaly degrades model accuracy once normal traffic patterns resume.
4. **Adversarial and Outlier Exploits:** Automated retraining loops are susceptible to data manipulation and feedback loops, where model outputs influence incoming data distributions.

---

## 2. Drift Detection Architecture

The monitoring module (`monitoring/monitor_drift.py`) uses **Evidently AI** to perform statistical hypothesis testing comparing reference distributions (training baseline) against live production inference windows:

```
Reference Ingested Train Data (Baseline) ◄───┐
                                              ├──► Evidently Statistical Audit
Current Production Logged Requests Window ───┘           │
                                                         ▼
                                       ┌───────────────────────────────────┐
                                       │ 1. Interactive HTML Report        │
                                       │    (artifacts/monitoring/*.html)  │
                                       │ 2. Structured Alert JSON Manifest │
                                       │    (artifacts/monitoring/         │
                                       │     drift_alert.json)             │
                                       └───────────────────────────────────┘
                                                         │
                                                         ▼
                                          Multi-Tier Severity Evaluation
                                         [NO_DRIFT / INFO / WARN / CRIT]
                                                         │
                                                         ▼
                                          Human-in-the-Loop Review SOP
```

### Statistical Methods Employed:
- **Numerical Features** (`tenure`, `MonthlyCharges`, `TotalCharges`): Two-sample Kolmogorov-Smirnov (K-S) test or Wasserstein distance.
- **Categorical Features** (`Contract`, `InternetService`, `PaymentMethod`, etc.): Chi-Square goodness-of-fit test or Jensen-Shannon divergence.
- **Target Distribution Drift:** Drift testing on historical vs. live predicted probability distributions.

---

## 3. Drift Severity Classification

The pipeline evaluates drift metrics against multi-tiered operational thresholds:

| Severity Level | Trigger Criteria | Action Required |
|---|---|---|
| **NO_DRIFT** | 0 features drifted, no target drift | No action required. Operational health verified. |
| **INFO** | Drift share $< 10\%$, minor non-critical features | Logged in monitoring audit log. Reviewed during weekly cycle. |
| **WARNING** | Drift share $10\% - 30\%$, OR 1 key churn driver drifted | Notification sent to MLOps team channel. Triage within 24 hours. |
| **CRITICAL** | Drift share $\ge 30\%$, OR target drift detected, OR $\ge 2$ key churn drivers drifted | High-priority PagerDuty / Slack alert. Immediate human review triggered. |

### Key Churn Drivers Monitored with High Sensitivity:
- `Contract` (Month-to-month vs 1/2-year)
- `tenure` (Customer lifetime duration)
- `MonthlyCharges` (Price sensitivity)
- `InternetService` (Fiber optic vs DSL vs none)
- `TechSupport` (Support entitlement)

---

## 4. Alert Manifest Schema (`artifacts/monitoring/drift_alert.json`)

Whenever drift detection executes, a machine-readable JSON manifest is written:

```json
{
  "alert_id": "DRIFT-20261031_191904",
  "generated_at": "2026-10-31T19:19:04.429000Z",
  "severity": "CRITICAL",
  "drift_detected": true,
  "target_drift_detected": false,
  "drift_share": 0.50,
  "number_of_drifted_features": 2,
  "drifted_features": ["Contract", "MonthlyCharges"],
  "key_drivers_affected": ["Contract", "MonthlyCharges"],
  "html_report_path": "artifacts/monitoring/drift_report_20261031_191904.html",
  "reference_dataset": "artifacts/train.csv",
  "current_dataset": "artifacts/test.csv",
  "policy": {
    "automated_retrain": false,
    "automated_deploy": false,
    "action_required": "Follow human-in-the-loop review procedure in docs/DRIFT_RESPONSE_AND_ALERTING.md"
  }
}
```

---

## 5. Human-in-the-Loop Standard Operating Procedure (SOP)

When a `WARNING` or `CRITICAL` alert is triggered, the on-call engineer and Model Governance team follow this 5-step SOP:

```
[Alert Fired] ──► Phase 1: Triage & Audit HTML Report
                         │
                         ▼
                  Phase 2: Upstream Data Quality Verification
                         │
                         ▼
                  Phase 3: Label Maturity & Cohort Audit
                         │
                         ▼
                  Phase 4: Offline Shadow Model Benchmarking
                         │
                         ▼
                  Phase 5: Governance Review & Canary Rollout
```

### Phase 1: Triage & Distribution Inspection
1. Open the interactive HTML drift report referenced in `html_report_path`.
2. Inspect the p-values and visual empirical distributions for all flagged features.
3. Determine whether drift is isolated to specific categories or represents a systemic shift.

### Phase 2: Upstream Data Quality Verification
1. Rule out ETL, schema, and API bugs:
   - Check API payload logging for sudden spikes in null values or default fallback values.
   - Verify front-end application changes (e.g. new checkout UI or altered billing options).
2. If drift is caused by a telemetry defect:
   - File an upstream bug ticket.
   - Do NOT retrain the model.
   - Apply API validation or data transformation patches if necessary.

### Phase 3: Label Maturity & Cohort Audit
1. Verify whether mature ground-truth churn outcomes (60+ days) are available for the drifted cohort.
2. If ground-truth labels are unavailable, continue running the champion model with heightened monitoring.
3. Compute actual business conversion / churn rates from historical billing systems to validate if consumer behavior genuinely shifted.

### Phase 4: Offline Shadow Model Benchmarking
1. In an isolated feature branch, ingest the newly labeled dataset containing the drifted cohort.
2. Run the complete training pipeline (`./reproduce.sh`).
3. Evaluate candidate models on both the historical test split and the new cohort split.
4. Verify that the challenger model achieves statistically significant ROC-AUC lift over the champion model and baselines without regressing on existing customer segments.

### Phase 5: Governance Sign-off & Canary Rollout
1. Submit an evaluation report (`artifacts/experiment_manifest.json`) to the Model Governance Committee.
2. Upon approval, promote the candidate model to `Staging`.
3. Deploy to production using **Canary (10% traffic)** or **Blue-Green** deployment on AWS ECS.
4. Verify `/health` probes and latency metrics. If error rates or drift metrics anomaly occurs, execute instant rollback using `/api/promote-model` or ECS task reversion.
