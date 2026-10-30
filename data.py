# data.py
# Loads the SMS dataset and splits it into training and test sets.
# Both the machine learning model and the deep learning model use this
# file, so they are compared on exactly the same data.

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

FOLDER = Path(__file__).resolve().parent


def load_split():
    """Return X_train, X_test, y_train, y_test (labels are 'ham' or 'spam')."""
    df = pd.read_csv(FOLDER / "spam_data.csv", encoding="latin-1", usecols=[0, 1])
    df.columns = ["label", "text"]
    df = df.dropna().drop_duplicates().reset_index(drop=True)

    return train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )
