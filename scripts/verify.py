import joblib

# Paths to the saved model and scaler
model_path = "/home/ksamderi/stock/technical_analysis/model_outputs/svm_model.pkl"
scaler_path = "/home/ksamderi/stock/technical_analysis/model_outputs/scaler.pkl"

# Load the model and scaler
svm_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

print("SVM model loaded successfully!")
print("Scaler loaded successfully!")
