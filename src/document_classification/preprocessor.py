import os
import json
import logging
import re

from document_classification.utils import load_json, wrap_text


# Logger
ml_logger = logging.getLogger("ml_logger")


class Preprocessor(object):
    def __init__(self, lower, char_level, filters):

        # Cleaning parameters
        self.lower = lower
        self.char_level = char_level
        self.filters = filters

    def clean(self, text):
        """Basic text preprocessing."""

        # Case sensitive
        if self.lower:
            text = " ".join(token.lower() for token in text.split(" "))

        # Split into tokens
        if self.char_level:
            text = " ".join(token for token in text)
        else:
            text = " ".join(token for token in text.split(" "))

        # Filter
        for token in self.filters:
            text = text.replace(token, " ")

        # Remove leading and trailing spaces
        text = text.strip()

        return text

    def clean_df(self, df):

        # Clean inputs
        df.X = df.X.apply(func=self.clean)

        wrap_text("Preprocessed data")
        print (df.head(5))
        return df

    @classmethod
    def load(cls, filepath):
        contents = load_json(filepath)
        return cls(**contents)

    def save(self, filepath):
        with open(filepath, "w") as fp:
            json.dump(self.__dict__, fp, indent=4)


