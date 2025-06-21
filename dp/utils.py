import pandas as pd
from sklearn.model_selection import train_test_split


def load_client_data(client_id):
    """
    Load data for a specific client, split into train/test.
    Args: client_id (int): ID of the client (1-5).
    Returns: X_train, X_test, y_train, y_test (Features and labels split).
    """
    df = pd.read_csv(f"./data/clients/client_{client_id}.csv")
    X = df.drop("Readmitted", axis=1).values  # Drop the label column
    y = df["Readmitted"].values  # Extract the label column
    return train_test_split(X, y, test_size=0.2, random_state=42)


def get_model():
    """
    Initialize a logistic regression model.
    Returns: sklearn.linear_model.LogisticRegression: The model.
    """
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(solver="lbfgs", max_iter=1000)
