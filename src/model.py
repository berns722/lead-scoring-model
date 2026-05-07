# Model training functions

from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier


def train_decision_tree(preprocessor, X_train, y_train, random_state=1):
    """
    Builds a pipeline with a DecisionTreeClassifier and trains it.

    Parameters
    ----------
    preprocessor : sklearn transformer
        Your preprocessing pipeline (e.g., ColumnTransformer)
    X_train : array-like or DataFrame
        Training features
    y_train : array-like
        Training labels
    random_state : int
        Seed for reproducibility

    Returns
    -------
    model : sklearn.pipeline.Pipeline
        Trained pipeline model
    """
    dtree = DecisionTreeClassifier(random_state=random_state)

    model = make_pipeline(preprocessor, dtree)

    model.fit(X_train, y_train)

    return model
