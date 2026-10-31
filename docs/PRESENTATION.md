# PROJECT DEFENSE: END-TO-END MLOPS CUSTOMER CHURN PIPELINE

**Thesis Defense Presentation Slides**  
*Academic Year: 2026*  

---

## SLIDE 1: Title & Overview
* **Title:** End-to-End MLOps Customer Churn Prediction and Drift Auditing
* **Sub-title:** An Orchestrated, Reproducible ML Pipeline from Raw Ingestion to Production Monitoring
* **Presenter:** Final Year Project Candidate
* **Key Topics Covered:**
  * Modular ML Pipelines (DVC)
  * Experiment Tracking (MLflow)
  * Containerization & Serving (FastAPI, Docker)
  * CI/CD (GitHub Actions, AWS ECR/ECS Fargate)
  * Distribution Audit (Evidently AI Drift Analysis)

---

## SLIDE 2: Business & Technical Context
* **The Telecom Churn Challenge:**
  * Retention is 5x cheaper than acquiring new customers.
  * Customer churn directly impacts revenue stability.
* **Why Traditional ML Fails in Production:**
  * Siloed notebook development makes pipelines hard to reproduce.
  * Environment changes lead to deployment bugs ("it worked on my machine").
  * Concept/data drift causes models to silently lose accuracy in production.
* **The Solution:** A unified MLOps pipeline that automates training, versioning, serving, and continuous validation.

---

## SLIDE 3: System Architecture
* **Orchestration & Serving Topology:**
  * **Data & Pipelines:** Orchestrated via DVC stages, separating concerns into discrete, reproducible modules.
  * **Experiment Tracker:** MLflow tracks models, records hyperparameters, and serves as our Model Registry.
  * **Serving Core:** FastAPI application running inside a Docker container.
  * **Deployment Engine:** AWS ECS Fargate hosting the API behind an Application Load Balancer.
  * **Monitoring Core:** Evidently AI checks incoming traffic for distribution drift.

---

## SLIDE 4: The Modular Pipeline (DVC Stages)
* **Stage 1: Data Ingestion**
  * Downloads raw customer data and performs a stratified 80-20 train/test split.
* **Stage 2: Schema Validation**
  * Validates data splits against `schema/schema.yaml` specifications to check columns, types, and missing values.
* **Stage 3: Data Transformation**
  * Imputes missing values, normalizes numerical features via StandardScaler, and encodes categorical columns via OneHotEncoder.
* **Stage 4: Hyperparameter Tuning**
  * Runs 3-fold cross-validated grid search across Logistic Regression, Random Forest, and XGBoost.
* **Stage 5: Evaluation & Registration**
  * Computes final classification metrics and registers the champion model in MLflow.

---

## SLIDE 5: Experimental Results
* **Algorithm Comparison Metrics:**
  * XGBoost outperformed other models with a validation ROC-AUC score of **0.8459**.
  * Logistic Regression followed closely with a score of **0.8412**.
  * Random Forest achieved a score of **0.8407**.
* **Classification Matrix (XGBoost on Test Set):**
  * Accuracy: **80.55%**
  * Precision: **66.56%**
  * Recall: **53.74%**
  * F1 Score: **59.47%**

---

## SLIDE 6: Feature Importances
* **Key Indicators of Customer Churn:**
  1. **Contract Format (Month-to-month):** Month-to-month contracts are the strongest predictor of churn.
  2. **Tenure Length:** Customers with a tenure of under 6 months show significantly higher churn rates.
  3. **Internet Service (Fiber Optic):** Subscribers on fiber optic connections churn at higher rates.
  4. **Tech Support Availability:** Customers without technical support show increased churn risk.
  5. **Monthly Charges:** High monthly bills correlate with higher churn probability.

---

## SLIDE 7: Web Application & Dashboard Demo
* **Responsive Single-Page Interface:**
  * **Executive Dashboard:** Live metrics, active model details, Great Expectations validations card, and training controls.
  * **Interactive Model Promotion Sandbox:** Promote candidate models (LR, RF, XGBoost) to production instantly, dynamic recalculating metrics on the fly.
  * **Single Profile Predictor + XAI:** Fill customer profiles or use presets to calculate probability. Renders local feature impact attribution (XAI waterfall chart) and the AI Retention Advisor recommendations.
  * **Batch Analytics:** Drag-and-drop CSV uploads, cohort metrics, and churn split donut charts.
  * **Performance Logs Console:** Monospace terminal displaying live logging.

---

## SLIDE 8: Deployment & CI/CD Pipeline
* **GitHub Actions CI/CD Pipeline:**
  * Triggers on merges to `main`.
  * Runs automated unit tests and code validation.
  * Builds a Docker image, pushes it to Amazon ECR, and deploys it to AWS ECS Fargate.
* **Cloud Infrastructure Highlights:**
  * **AWS ECS Fargate:** Serverless container deployment eliminates server management.
  * **Application Load Balancer:** Ensures high availability.
  * **AWS Secrets Manager:** Injects environment variables securely at runtime.

---

## SLIDE 9: Production Auditing (Evidently AI)
* **Solving Silent Model Degradation:**
  * Compares production data splits against baseline training distributions.
  * Identifies changes in data distributions (data drift) and model target predictions (target drift).
  * Automatically generates diagnostic reports, which can be viewed directly within the web dashboard.
  * Multi-tier alerting (`NO_DRIFT`, `INFO`, `WARNING`, `CRITICAL`) with strict guardrails: zero automatic redeployments.
  * Triggers documented human-in-the-loop review SOP accounting for 30-60 day label lag.

---

## SLIDE 10: Summary & Key Takeaways
* **Project Deliverables:**
  * A reproducible ML pipeline orchestrated via DVC and tracked by MLflow.
  * Cryptographic artifact and dataset versioning, ROC-AUC benchmarking against dual baselines.
  * Rigorous test suite (32 tests) covering schema changes, imbalance, input validation, and deployment rollback.
  * Containerized FastAPI app with Explainable AI (XAI), What-If recommendation sandbox, and Great Expectations data hygiene assertions.
  * Automated cloud deployment and monitoring with deployment rollback safety.
* **Future Scope:**
  * Governed shadow retraining workflows with human-in-the-loop review.
  * Integrating a real-time feature store (e.g. Feast) for feature consistency.

