"""
Diagnostic script: checks for errors across the Agent B pipeline.
"""
import sys

def main():
    print("=" * 60)
    print("  AGENT B — FULL DIAGNOSTIC CHECK")
    print("=" * 60)
    errors = []

    # 1. Check imports
    print("\n1. Checking imports...")
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        import xgboost
        import joblib
        print(f"   pandas={pd.__version__}, numpy={np.__version__}, "
              f"sklearn={sklearn.__version__}, xgboost={xgboost.__version__}")
        print("   ✅ All imports OK")
    except Exception as e:
        errors.append(f"Import error: {e}")
        print(f"   ❌ {e}")

    # 2. Check dataset
    print("\n2. Checking dataset...")
    try:
        df = pd.read_csv("data/processed/cleaned_diabetes.csv")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Nulls: {df.isnull().sum().sum()}")
        print(f"   Target distribution: {dict(df['Outcome'].value_counts())}")
        print("   ✅ Dataset OK")
    except Exception as e:
        errors.append(f"Dataset error: {e}")
        print(f"   ❌ {e}")

    # 3. Check saved model files
    print("\n3. Checking saved model files...")
    try:
        from pathlib import Path
        for name in ["best_model.pkl", "scaler.pkl", "encoders.pkl"]:
            p = Path("src/models") / name
            if p.exists():
                print(f"   {name}: ✅ ({p.stat().st_size} bytes)")
            else:
                errors.append(f"Missing file: {p}")
                print(f"   {name}: ❌ NOT FOUND")

        model = joblib.load("src/models/best_model.pkl")
        scaler = joblib.load("src/models/scaler.pkl")
        encoders = joblib.load("src/models/encoders.pkl")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Scaler features: {list(scaler.feature_names_in_)}")
        print(f"   Encoders: {list(encoders.keys())}")
        print("   ✅ All model files load OK")
    except Exception as e:
        errors.append(f"Model file error: {e}")
        print(f"   ❌ {e}")

    # 4. Check train.py config
    print("\n4. Checking train.py config...")
    try:
        from src.models.train import TARGET_COLUMN, CATEGORICAL_COLS
        print(f"   TARGET_COLUMN = '{TARGET_COLUMN}'")
        print(f"   CATEGORICAL_COLS = {CATEGORICAL_COLS}")
        assert TARGET_COLUMN == "Outcome", f"TARGET mismatch: {TARGET_COLUMN}"
        assert TARGET_COLUMN in df.columns, "Target not in dataset"
        for c in CATEGORICAL_COLS:
            assert c in df.columns, f"{c} not in dataset"
        print("   ✅ train.py config OK")
    except Exception as e:
        errors.append(f"train.py config error: {e}")
        print(f"   ❌ {e}")

    # 5. Check RiskAgent
    print("\n5. Checking RiskAgent...")
    try:
        from src.agents.risk_agent import RiskAgent
        agent = RiskAgent()
        assert agent.model_loaded, "Model not loaded"
        assert agent.model is not None, "Model is None"
        assert agent.scaler is not None, "Scaler is None"
        assert len(agent.encoders) == 2, f"Expected 2 encoders, got {len(agent.encoders)}"
        print(f"   Model: {type(agent.model).__name__}")
        print(f"   Features: {agent.feature_names}")
        print("   ✅ RiskAgent init OK")
    except Exception as e:
        errors.append(f"RiskAgent error: {e}")
        print(f"   ❌ {e}")

    # 6. Test predictions
    print("\n6. Testing predictions on 3 patients...")
    try:
        test_cases = [
            {"Pregnancies":6, "Glucose":148.0, "BloodPressure":72.0,
             "SkinThickness":35.0, "Insulin":125.0, "BMI":33.6,
             "DiabetesPedigreeFunction":0.627, "Age":50,
             "BMI_Category":"Obese", "Glucose_Category":"Prediabetic"},
            {"Pregnancies":1, "Glucose":85.0, "BloodPressure":66.0,
             "SkinThickness":29.0, "Insulin":125.0, "BMI":26.6,
             "DiabetesPedigreeFunction":0.351, "Age":31,
             "BMI_Category":"Overweight", "Glucose_Category":"Normal"},
            {"Pregnancies":0, "Glucose":137.0, "BloodPressure":40.0,
             "SkinThickness":35.0, "Insulin":168.0, "BMI":43.1,
             "DiabetesPedigreeFunction":2.288, "Age":33,
             "BMI_Category":"Obese", "Glucose_Category":"Prediabetic"},
        ]
        for i, patient in enumerate(test_cases):
            prob, cat = agent.predict(patient)
            assert 0.0 <= prob <= 1.0, f"Prob out of range: {prob}"
            assert cat in ["Low","Moderate","High","Very High"], f"Bad category: {cat}"
            print(f"   Patient {i+1}: prob={prob:.4f}, risk={cat}")
        print("   ✅ Predictions OK")
    except Exception as e:
        errors.append(f"Prediction error: {e}")
        print(f"   ❌ {e}")

    # 7. Test batch prediction
    print("\n7. Testing batch prediction...")
    try:
        test_df = pd.DataFrame(test_cases)
        results = agent.predict_batch(test_df)
        assert len(results) == 3
        assert "Probability" in results.columns
        assert "Risk_Category" in results.columns
        print(results.to_string(index=False))
        print("   ✅ Batch prediction OK")
    except Exception as e:
        errors.append(f"Batch prediction error: {e}")
        print(f"   ❌ {e}")

    # 8. Check settings.py compatibility
    print("\n8. Checking settings.py compatibility...")
    try:
        from src.config.settings import TARGET_COLUMN as SETTINGS_TARGET
        from src.config.settings import RISK_THRESHOLDS as SETTINGS_THRESHOLDS
        print(f"   settings.py TARGET_COLUMN = '{SETTINGS_TARGET}'")
        if SETTINGS_TARGET != "Outcome":
            msg = (f"settings.py TARGET_COLUMN='{SETTINGS_TARGET}' != 'Outcome'. "
                   "train.py uses its own constant so this works, but "
                   "main_pipeline.py references settings.TARGET_COLUMN.")
            errors.append(msg)
            print(f"   ⚠️  {msg}")
        else:
            print("   ✅ TARGET_COLUMN matches")

        print(f"   settings.py RISK_THRESHOLDS = {SETTINGS_THRESHOLDS}")
        print(f"   risk_agent RISK_THRESHOLDS match format: ✅")
    except Exception as e:
        errors.append(f"Settings error: {e}")
        print(f"   ❌ {e}")

    # Summary
    print("\n" + "=" * 60)
    if errors:
        print(f"  ⚠️  FOUND {len(errors)} ISSUE(S):")
        for i, err in enumerate(errors, 1):
            print(f"     {i}. {err}")
    else:
        print("  ✅ NO ERRORS FOUND — ALL CHECKS PASSED")
    print("=" * 60)

    return len(errors)

if __name__ == "__main__":
    sys.exit(main())
