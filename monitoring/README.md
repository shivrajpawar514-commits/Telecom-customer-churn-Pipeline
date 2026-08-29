# Monitoring

This project separates monitoring into two concerns:

## 1. Model / data drift (`monitor_drift.py`)
Uses [Evidently](https://www.evidentlyai.com/) to compare a reference dataset
(training data) against a current window of production data (logged
inference requests) and produces an HTML drift report.

Run manually:
```bash
python monitoring/monitor_drift.py
```

In production, schedule this as:
- An AWS EventBridge rule triggering a Lambda or Fargate task nightly, or
- A step in Airflow/Step Functions after enough requests accumulate.

Store `artifacts/monitoring/*.html` in S3 for historical reports, and push the
`dataset_drift` boolean to CloudWatch as a custom metric to alarm on.

## 2. Service health / performance
The FastAPI app exposes `GET /health` for liveness checks (used by the
Dockerfile HEALTHCHECK, ECS task health checks, and an ALB target group).

Recommended CloudWatch setup once deployed to ECS:
- **Container Insights** enabled on the ECS cluster for CPU/memory.
- **ALB target group health checks** hitting `/health`.
- **Custom metric** for prediction latency + request count, emitted via
  `boto3.client("cloudwatch").put_metric_data(...)` in `app.py` middleware.
- **CloudWatch Alarms** -> SNS topic -> email/Slack for: high 5xx rate,
  high p95 latency, drift detected == true.
