#Data preprocessing funcitons
from IPython.display import display

def head_and_tail(df, n=5):
    print("Head:")
    display(df.head(n))
    print("\nTail:")
    display(df.tail(n))