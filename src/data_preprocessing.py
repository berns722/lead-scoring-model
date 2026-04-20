#Data preprocessing funcitons
from IPython.display import display

def head_and_tail(df, n=5):
    print("Head:")
    display(df.head(n))
    print("\nTail:")
    display(df.tail(n))

def preprocess_data(df):
    df = df.copy()
    df = df.drop(columns=['ID'])  # Drop ID column
    return df
