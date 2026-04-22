#Data preprocessing functions

def preprocess_data(df):
    df = df.copy()
    df = df.drop(columns=['ID'])  # Drop ID column
    return df

