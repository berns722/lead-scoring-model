# Data preprocessing functions


def preprocess_data(df):
    """
    Prepares the dataset for modeling.
    Current steps:
        - Drop ID column (no predictive value)
    Future steps:
        - Encoding
        - Scaling
        - Imputation
    """
    df = df.copy()
    df = df.drop(columns=["ID"])
    return df
