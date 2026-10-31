# Secrets Management, Container Health & Deployment Safety Guide

This document details security guardrails, secret management policies, container healthcheck configurations, and deployment rollback mechanisms implemented in the Telecom Customer Churn prediction service.

---

## 1. Secrets Management Architecture

To prevent credential leakage and unauthorized cloud access, credentials and secrets are **strictly decoupled from code, Docker images, and git repositories**.

```
┌─────────────────────────────────────────────────────────────┐
│                       SECRET SOURCES                        │
├──────────────────────────────┬──────────────────────────────┤
│ Development / Local Testing  │ Production / Cloud (AWS)     │
├──────────────────────────────┼──────────────────────────────┤
│ .env (untracked)             │ AWS Secrets Manager          │
│ .env.example (placeholders)  │ AWS SSM Parameter Store      │
│                              │ GitHub Encrypted Secrets     │
└──────────────┬───────────────┴──────────────┬───────────────┘
               │                              │
               ▼                              ▼
      Local Python Runtime            ECS Task Definition
      (dotenv via os.environ)       (Runtime Secret Injection)
                                              │
                                              ▼
                                     Docker Container Process
                                    (Zero secrets in image layers)
```

### Key Principles:
1. **Never Bake Secrets into Images:** The `Dockerfile` does not copy or contain `.env`, `.pem`, `.key`, or credentials. Build layers remain completely free of sensitive secrets.
2. **Git & Docker Exclusion Guardrails:**
   - `.gitignore` explicitly ignores `.env`, `venv/`, `.venv/`, `artifacts/*.pkl`, `*.pem`, `*.key`, `credentials`.
   - `.dockerignore` excludes `.env`, `tests/`, `*.md`, `notebooks/`, and secret patterns.
3. **Automated Secrets Safety Tests:**
   The test suite includes `tests/test_secrets_safety.py`, which validates on every CI run:
   - `test_no_hardcoded_aws_keys_in_source`: Scans all codebase files for regex pattern `AKIA[0-9A-Z]{16}`.
   - `test_env_file_not_tracked_in_git`: Verifies `.env` is never indexed in git.
   - `test_gitignore_contains_sensitive_patterns`: Validates exclusion rules.
   - `test_dockerignore_contains_sensitive_patterns`: Validates Docker context rules.
   - `test_env_example_uses_placeholders`: Ensures only dummy placeholders (`changeme`) exist in sample files.

---

## 2. Cloud Secrets Injection (AWS ECS / Fargate)

In production, ECS tasks receive credentials at container startup via ECS Task Execution Role IAM policies.

### ECS Task Definition Configuration Example:
```json
{
  "containerDefinitions": [
    {
      "name": "churn-api",
      "image": "123456789012.dkr.ecr.ap-south-1.amazonaws.com/telecom-churn-pipeline:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "hostPort": 8000,
          "protocol": "tcp"
        }
      ],
      "essential": true,
      "secrets": [
        {
          "name": "MLFLOW_TRACKING_URI",
          "valueFrom": "arn:aws:secretsmanager:ap-south-1:123456789012:secret:churn/mlflow-uri"
        },
        {
          "name": "DATA_SOURCE_URL",
          "valueFrom": "arn:aws:ssm:ap-south-1:123456789012:parameter/churn/data-url"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/churn-api",
          "awslogs-region": "ap-south-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

---

## 3. Container Health Checks & Liveness Probes

The FastAPI application implements a dedicated health check endpoint at `GET /health`.

### Health Check Response:
```json
{
  "status": "healthy",
  "service": "telecom-churn-api",
  "version": "1.5.0",
  "model_loaded": true,
  "preprocessor_loaded": true
}
```

### Dockerfile Health Check:
```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
```

If the active model file (`artifacts/model.pkl`) or preprocessor (`artifacts/preprocessor.pkl`) is unreadable, `/health` reports `"status": "degraded"`, prompting the ALB target group and ECS orchestrator to mark the container unhealthy and prevent routing production traffic to it.

---

## 4. Model Loading Validation & Dynamic Rollback

### Model Loading Test Suite (`tests/test_deployment_and_serving.py`):
1. **Model Loading Success:** Validates that `artifacts/model.pkl` and `artifacts/preprocessor.pkl` can be unpickled and expose `.predict()` and `.transform()` methods.
2. **Missing Artifact Handling:** Verifies that attempting to load a missing model raises a structured `FileNotFoundError` rather than crashing silently.
3. **Container Compatibility:** Simulates Docker healthcheck probes to verify response codes and response times ($< 50$ ms).

### Deployment Rollback Mechanism:
If a newly deployed model exhibits degraded behavior, latency spikes, or prediction drift, operators can trigger an instant rollback without waiting for a new image rebuild:
1. Candidate models evaluated during training (`model_LogisticRegression.pkl`, `model_RandomForest.pkl`, `model_XGBoost.pkl`) are preserved in `artifacts/`.
2. Operators invoke `POST /api/promote-model` with `{"model_name": "RandomForest"}` (or previous champion).
3. The API hot-reloads the selected estimator in memory, verified by `test_deployment_rollback`.
4. At the infrastructure level, ECS task revisions allow one-click rollback to the previous stable task definition:
   ```bash
   aws ecs update-service --cluster churn-cluster --service churn-service --task-definition churn-api:<PREVIOUS_REVISION>
   ```
