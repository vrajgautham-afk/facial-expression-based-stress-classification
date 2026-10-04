import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ---------------------------------------
# Load dataset
# ---------------------------------------

data = pd.read_csv("stress_data.csv")

print("Dataset:")
print(data.head())

# ---------------------------------------
# Separate features and target
# ---------------------------------------

X = data.drop("stress", axis=1)
y = data["stress"]

# ---------------------------------------
# Split dataset
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    
)

# ---------------------------------------
# Build ML pipeline
# ---------------------------------------

model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
    )
])

# ---------------------------------------
# Train
# ---------------------------------------

model.fit(X_train, y_train)

# ---------------------------------------
# Test
# ---------------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)

# ---------------------------------------
# Save model
# ---------------------------------------

joblib.dump(
    model,
    "stress_model.pkl"
)

print("\nModel saved as stress_model.pkl")