# Model & Data Drift Monitoring

This directory contains the drift auditing system for the deployed telecom customer churn prediction service.

## 1. Drift Monitoring Engine (`monitor_drift.py`)
Uses [Evidently](https://www.evidentlyai.com/) to compare a reference training dataset against a current window of live production inference requests.

### Artifacts Generated:
1. **Interactive HTML Report:** Written to `artifacts/monitoring/drift_report_<TIMESTAMP>.html` for visual empirical distribution inspection and statistical p-values.
2. **Machine-Readable Alert Manifest:** Written to `artifacts/monitoring/drift_alert.json` containing severity tiers, drifted feature lists, and policy constraints.

### Run Manually:
```bash
python monitoring/monitor_drift.py
```
Or via Makefile / Reproducibility runner:
```bash
make drift
# or
./reproduce.sh
```

---

## 2. Policy Guardrail: Zero Automated Retraining on Drift

> [!CAUTION]
> **DO NOT AUTOMATICALLY RETRAIN OR DEPLOY A MODEL JUST BECAUSE DRIFT WAS DETECTED.**
>
> In subscription churn, ground truth labels take **30 to 60 days** to mature. Auto-retraining on live inference traffic before labels mature forces the pipeline to train on unverified pseudo-labels or transient noise. Furthermore, upstream data corruption, schema drift, or temporary promotional spikes can poison the model distribution.
>
> Always follow the human-in-the-loop Standard Operating Procedure documented in [`docs/DRIFT_RESPONSE_AND_ALERTING.md`](../docs/DRIFT_RESPONSE_AND_ALERTING.md).

---

## 3. Drift Severity Classification

- **NO_DRIFT:** 0 drifted features. Normal operational state.
- **INFO:** Drift share $< 10\%$, non-critical features. Logged for weekly audit.
- **WARNING:** Drift share $10\% - 30\%$, OR 1 key churn driver drifted. 24h triage required.
- **CRITICAL:** Drift share $\ge 30\%$, OR target drift, OR $\ge 2$ key churn drivers drifted (`Contract`, `tenure`, `MonthlyCharges`, `InternetService`, `TechSupport`). Immediate engineer triage and review required.

---

## 4. Production Scheduling Pattern
In AWS production environments:
1. Run `monitor_drift.py` as a scheduled AWS EventBridge rule triggering an ECS task or Lambda nightly on logged inference requests.
2. Publish `artifacts/monitoring/*.html` to a secure internal Amazon S3 bucket.
3. Emit drift metrics (`drift_share`, `number_of_drifted_columns`, `severity`) to CloudWatch Custom Metrics.
4. Route `CRITICAL` alerts via Amazon SNS to the MLOps team Slack/PagerDuty channel to initiate the human review SOP.
