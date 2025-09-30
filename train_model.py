# train_model.py
# Train a RandomForest on the Iris dataset and save the model to iris_rf_model.joblib

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

OUTPUT_MODEL_PATH = "iris_rf_model.joblib"

def main():
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)

    print("Accuracy:", accuracy_score(y_test, preds))
    print("Classification report:\n", classification_report(y_test, preds, target_names=iris.target_names))

    # Save the model
    joblib.dump({
        "model": clf,
        "feature_names": list(X.columns),
        "target_names": list(iris.target_names)
    }, OUTPUT_MODEL_PATH)

    print(f"Saved trained model to {OUTPUT_MODEL_PATH}")

if __name__ == "__main__":
    main()
