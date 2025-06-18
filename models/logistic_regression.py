import json
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline  





def load_dataset(path, sample=None):
    """
    Load the preprocessed dataset from a CSV file.

    Args:
        path: Path to the preprocessed CSV file.
        sample: (int, optional) If provided, randomly sample this number of rows.

    Returns:
        The loaded dataset as a pandas DataFrame.
    """
    df = pd.read_csv(path)

    if sample:
        df = df.sample(n=sample, random_state=42).reset_index(drop=True)

    return df


def split_features_labels(df):
    """
    Split the DataFrame into features (X) and label (y).

    Args:
        df: The preprocessed dataset (pandas DataFrame).

    Returns:
        X: Features as a NumPy array.
        y: Labels as a NumPy array.
    """
    X = df.drop("Readmitted", axis=1).values
    y = df["Readmitted"].values

    return X, y


# def train_logistic_regression(X_train, y_train):
#     """
#     Train a logistic regression model on the given training data.

#     Args:
#         X_train: Training features (NumPy array).
#         y_train: Training labels (NumPy array).

#     Returns:
#         Trained LogisticRegression model.
#     """
#     model = LogisticRegression(
#         solver="liblinear",  # good for small-to-medium datasets
#         max_iter=20000,       # higher iteration cap for convergence
#         random_state=42,
#         C=0.1,
#         class_weight="balanced"
#     )
#     model.fit(X_train, y_train)

#     print("Logistic regression training complete.")
#     return model


def train_logistic_regression(X_train, y_train):
    """
    Train a logistic regression model on the given training data.
    Applies SMOTE to rebalance the training set.

    Args:
        X_train: Training features (NumPy array).
        y_train: Training labels (NumPy array).

    Returns:
        Trained LogisticRegression model.
    """
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"SMOTE applied: {sum(y_resampled==1)} positives, {sum(y_resampled==0)} negatives")

    # Train the model
    model = LogisticRegression(
        solver="liblinear",
        max_iter=20000,
        random_state=42,
        C=0.1,
        class_weight="balanced"  # optional: you can remove this now, SMOTE already balances
    )
    model.fit(X_resampled, y_resampled)

    print("✅ Logistic regression training complete.")
    return model




def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test set.

    Args:
        model: Trained logistic regression model.
        X_test: Test features (NumPy array).
        y_test: True test labels (NumPy array).

    Returns:
        Dictionary with accuracy, precision, recall, and AUC scores.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # for AUC

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_proba)
    }

    print("Evaluation complete.")
    for k, v in metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")

    return metrics




def save_metrics(metrics, filename="evaluation/metrics.json"):
    """
    Save evaluation metrics to a JSON file.

    Args:
        metrics: Dictionary of evaluation results.
        filename: Path to output JSON file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics saved to: {filename}")




def plot_roc_curve(model, X_test, y_test, filename="evaluation/roc_curve.png"):
    """
    Plot and save the ROC curve.

    Args:
        model: Trained classifier with `predict_proba`.
        X_test: Test features.
        y_test: True labels.
        filename: Output image path.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    plt.figure()
    plt.plot(fpr, tpr, label="ROC curve (AUC = {:.2f})".format(roc_auc_score(y_test, y_proba)))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.grid(True)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()

    print(f"ROC curve saved to: {filename}")
    
    
def grid_search_logistic_regression(X, y):
    """
    Run grid search over logistic regression hyperparameters using SMOTE + polynomial interaction features.

    Args:
        X: Features
        y: Labels

    Returns:
        Best trained model and grid search object
    """
    # Column names (used for ColumnTransformer)
    numeric_cols = [0, 1]  # Assuming "Age" and "Billing Amount" are the first two columns

    # Polynomial + Scaling only on numeric columns
    preprocessor = ColumnTransformer(transformers=[
        ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False), numeric_cols),
        ('scale', StandardScaler(), numeric_cols)
    ], remainder='passthrough')  # passthrough for one-hot encoded features

    pipeline = ImbPipeline(steps=[
        ('preprocess', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('logreg', LogisticRegression(solver='liblinear', max_iter=10000, random_state=42))
    ])

    param_grid = {
        'logreg__C': [0.01, 0.1, 1, 10],
        'logreg__class_weight': [None, 'balanced'],
        'logreg__penalty': ['l2']
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=5,
        verbose=2,
        n_jobs=-1
    )

    print("****Running grid search with polynomial features...****")
    grid_search.fit(X, y)

    print("\n*******Grid search complete.*******")
    print("Best parameters:", grid_search.best_params_)
    print("Best ROC AUC score:", grid_search.best_score_)

    return grid_search.best_estimator_, grid_search




    

if __name__ == "__main__":
    # Load
    df = load_dataset("data/processed/full_preprocessed.csv")
    
    # Prepare
    X, y = split_features_labels(df)
    
    print("Label distribution:", pd.Series(y).value_counts())
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train with grid search (includes SMOTE)
    model, grid = grid_search_logistic_regression(X_train, y_train)

    # # Evaluate
    metrics = evaluate_model(model, X_test, y_test)

    # # Save results
    save_metrics(metrics)
    plot_roc_curve(model, X_test, y_test)

