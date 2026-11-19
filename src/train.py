
# ==========================================
# MLOps Assignment 2
# Track Machine Learning Experiments Using MLflow
# ==========================================

# Importing required libraries

import os
import mlflow
import mlflow.sklearn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


# ==========================================
# STEP 1: Load the Iris Dataset
# ==========================================

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["target"] = iris.target

print("First five rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTarget Classes:")
print(iris.target_names)


# ==========================================
# STEP 2: Preprocessing and Data Splitting
# ==========================================

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

print("\nTraining Class Distribution:")
print(y_train.value_counts())

print("\nTesting Class Distribution:")
print(y_test.value_counts())


# ==========================================
# STEP 3: Define Machine Learning Models
# ==========================================

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=200))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=3,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=3,
        random_state=42
    )
}

print("\nModels initialized successfully!")

for model_name in models:
    print(model_name)


# ==========================================
# STEP 4: Configure MLflow Experiment
# ==========================================

mlflow.set_experiment("Iris_Classification_Experiments")

print("\nMLflow experiment configured successfully!")

mlflow.sklearn.autolog(disable=True)


# ==========================================
# STEP 5: Train Models and Track Experiments
# ==========================================

os.makedirs("artifacts", exist_ok=True)

results = []

for model_name, model in models.items():

    with mlflow.start_run(run_name=model_name):

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test, y_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test, y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test, y_pred,
            average="weighted",
            zero_division=0
        )

        # Log parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("test_size", 0.20)
        mlflow.log_param("random_state", 42)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(6, 5))

        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=iris.target_names
        ).plot(cmap="Blues")

        plt.title(f"Confusion Matrix - {model_name}")
        plt.tight_layout()

        safe_name = model_name.replace(" ", "_")

        cm_path = f"artifacts/{safe_name}_confusion_matrix.png"

        plt.savefig(cm_path)
        plt.close()

        # Log confusion matrix artifact
        mlflow.log_artifact(cm_path)

        # Generate classification report
        report = classification_report(
            y_test,
            y_pred,
            target_names=iris.target_names
        )

        report_path = (
            f"artifacts/{safe_name}_classification_report.txt"
        )

        with open(report_path, "w") as file:
            file.write(report)

        # Log classification report artifact
        mlflow.log_artifact(report_path)

        # Log trained model using Cloudpickle
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format="cloudpickle"
        )

        # Store results for comparison
        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

        # Display results
        print(f"\nModel: {model_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

print("\nAll experiments completed successfully!")


# ==========================================
# STEP 6: Compare Model Performance
# ==========================================

results_df = pd.DataFrame(results)

print("\nModel Comparison:")
print(results_df.to_string(index=False))


# ==========================================
# STEP 7: Select the Best Model
# ==========================================

best_model_result = max(
    results,
    key=lambda x: (x["F1 Score"], x["Accuracy"])
)

best_model_name = best_model_result["Model"]

print("\nBest Performing Model:", best_model_name)
print("Best Accuracy:", best_model_result["Accuracy"])
print("Best F1 Score:", best_model_result["F1 Score"])


# ==========================================
# STEP 8: Register the Best Model
# ==========================================

best_model = models[best_model_name]

with mlflow.start_run(run_name="Best_Model_Registration"):

    model_info = mlflow.sklearn.log_model(
        sk_model=best_model,
        name="best_model",
        serialization_format="cloudpickle",
        registered_model_name="Best_Iris_Classification_Model"
    )

print("\nBest model registered successfully!")
print("Registered Model: Best_Iris_Classification_Model")
print("\nAssignment implementation completed successfully!")
