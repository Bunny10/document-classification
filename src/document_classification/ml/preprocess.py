import os
import re

def preprocess_sections(sections):
    """Basic text preprocessing.
    """
    text = " ".join(word.lower() for word in sections.split(" "))
    text = text.replace("\n", " ")
    text = re.sub(r"[^a-zA-Z_]+", r" ", text)
    text = text.strip()
    return text


def preprocess_data(df):
    """Preprocess dataframe."""
    df.X = df.X.apply(preprocess_sections)
    return df