import os
import sys
import socket
import json

def check_port(host, port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            s.connect((host, port))
            return True
    except Exception:
        return False

def print_header(title):
    print("=" * 60)
    print(f" {title} ".center(60, "="))
    print("=" * 60)

def main():
    print_header("ChurnOps MLOps Diagnostic Checklist")
    
    # 1. Environment Checks
    print("\n[1] Environment & Python Checks:")
    print(f"  - Python Version: {sys.version.split()[0]}")
    print(f"  - Working Directory: {os.getcwd()}")
    
    # 2. Package Check
    print("\n[2] Key Dependencies Check:")
    dependencies = ["fastapi", "uvicorn", "sklearn", "xgboost", "mlflow", "evidently", "multipart"]
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"  - {dep:<12}: OK")
        except ImportError:
            if dep == "multipart":
                try:
                    import libmultiparts
                    print(f"  - {dep:<12}: OK (custom namespace)")
                except ImportError:
                    print(f"  - {dep:<12}: FAILED (Install python-multipart)")
            else:
                print(f"  - {dep:<12}: FAILED (Missing)")

    # 3. Artifact Checks
    print("\n[3] Model Pipeline Artifacts Check:")
    required_files = {
        "artifacts/model.pkl": "Active serving model weights",
        "artifacts/preprocessor.pkl": "Data scaling/onehot pipeline",
        "artifacts/model_XGBoost.pkl": "XGBoost candidate weight file",
        "artifacts/model_RandomForest.pkl": "RandomForest candidate weight file",
        "artifacts/model_LogisticRegression.pkl": "LogisticRegression candidate weight file",
        "artifacts/metrics.json": "Active evaluation metrics payload",
        "artifacts/model_report.json": "MLflow model registry report",
        "artifacts/validation_report.json": "Dataset schema validation details",
        "artifacts/test.csv": "Evaluation split dataset"
    }
    
    all_artifacts_ok = True
    for filepath, desc in required_files.items():
        if os.path.exists(filepath):
            size_kb = os.path.getsize(filepath) / 1024
            print(f"  - {filepath:<35}: OK ({size_kb:.1f} KB) - {desc}")
        else:
            print(f"  - {filepath:<35}: MISSING - {desc}")
            all_artifacts_ok = False
            
    # 4. Pipeline Execution Checks
    if all_artifacts_ok:
        print("\n[4] Local Predict Pipeline Test:")
        try:
            import joblib
            import pandas as pd
            
            # Load preprocessor & model
            model = joblib.load("artifacts/model.pkl")
            prep = joblib.load("artifacts/preprocessor.pkl")
            
            # Test record
            test_row = pd.DataFrame([{
                "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
                "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",
                "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "Yes",
                "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "No",
                "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.35, "TotalCharges": 845.5
            }])
            
            transformed = prep.transform(test_row)
            if hasattr(transformed, "toarray"):
                transformed = transformed.toarray()
                
            pred = model.predict(transformed)[0]
            prob = model.predict_proba(transformed)[0][1]
            print(f"  - Load & Inference Pipeline Test: OK (Predicted: {pred}, Risk: {prob*100:.1f}%)")
        except Exception as e:
            print(f"  - Load & Inference Pipeline Test: FAILED (Error: {str(e)})")
    else:
        print("\n[4] Local Predict Pipeline Test: SKIPPED (Missing components)")

    # 5. Network Port listener checks
    print("\n[5] Server Network Bindings:")
    port_8000_ok = check_port("127.0.0.1", 8000)
    if port_8000_ok:
        print("  - Port 8000 (FastAPI Server)    : ACTIVE & LISTENING")
        print("\n  👉 Dashboard is fully available at: http://localhost:8000/")
    else:
        print("  - Port 8000 (FastAPI Server)    : INACTIVE (Run './venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000')")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
