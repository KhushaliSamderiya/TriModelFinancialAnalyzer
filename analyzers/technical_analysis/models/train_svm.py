import pandas as pd
import argparse
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# Argument parser for user input
parser = argparse.ArgumentParser(description="Train an SVM model for technical analysis.")
parser.add_argument("--train_file", type=str, required=True, help="Path to the training CSV file.")
parser.add_argument("--test_file", type=str, required=True, help="Path to the testing CSV file.")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained model and results.")
parser.add_argument("--kernel", type=str, default="rbf", help="SVM kernel type (linear, poly, rbf, sigmoid).")
parser.add_argument("--C", type=float, default=1.0, help="Regularization parameter.")
args = parser.parse_args()

# Ensure output directory exists
os.makedirs(args.output_dir, exist_ok=True)

# Load the data
print("Loading data...")
train_data = pd.read_csv(args.train_file)
test_data = pd.read_csv(args.test_file)

# Define features and target variable
features = ["Daily Return", "SMA_50", "SMA_200"]
target = "Adj Close"  # Replace with your target variable, e.g., price movement classification

# Assign labels for classification (Buy, Hold, Sell)
def assign_labels(row):
    if row["Daily Return"] > 0.02:  # Example threshold for 'Buy'
        return "Buy"
    elif row["Daily Return"] < -0.02:  # Example threshold for 'Sell'
        return "Sell"
    else:
        return "Hold"

train_data["Label"] = train_data.apply(assign_labels, axis=1)
test_data["Label"] = test_data.apply(assign_labels, axis=1)

# Encode labels
label_mapping = {"Buy": 0, "Hold": 1, "Sell": 2}
train_data["Label"] = train_data["Label"].map(label_mapping)
test_data["Label"] = test_data["Label"].map(label_mapping)

# Drop rows with missing or invalid labels
train_data.dropna(subset=features + ["Label"], inplace=True)
test_data.dropna(subset=features + ["Label"], inplace=True)

# Prepare training and testing sets
X_train = train_data[features]
y_train = train_data["Label"]
X_test = test_data[features]
y_test = test_data["Label"]

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the SVM model
print("Training SVM model...")
svm_model = SVC(kernel=args.kernel, C=args.C, probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Save the trained model and scaler
joblib.dump(svm_model, os.path.join(args.output_dir, "svm_model.pkl"))
joblib.dump(scaler, os.path.join(args.output_dir, "scaler.pkl"))
print(f"Model and scaler saved to {args.output_dir}")

# Evaluate the model
print("Evaluating model...")
y_pred = svm_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_mapping.keys())
conf_matrix = confusion_matrix(y_test, y_pred)

# Save evaluation metrics
with open(os.path.join(args.output_dir, "evaluation.txt"), "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(conf_matrix))

print(f"Evaluation complete. Results saved to {args.output_dir}")
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)
