# test_dependencies.py
try:
    import joblib
    print("✅ joblib available")
except ImportError as e:
    print(f"❌ joblib missing: {e}")

try:
    import numpy as np
    print("✅ numpy available")
    print(f"   Version: {np.__version__}")
except ImportError as e:
    print(f"❌ numpy missing: {e}")

try:
    from sklearn.preprocessing import LabelEncoder
    print("✅ scikit-learn available")
except ImportError as e:
    print(f"❌ scikit-learn missing: {e}")
