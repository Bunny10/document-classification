import os
import json
import logging
import re

from document_classification.utils import load_json, wrap_text


# Logger
ml_logger = logging.getLogger("ml_logger")


class Preprocessor(object):
    def __init__(self, input_feature, output_feature, split_level,
                 case_sensitive, allow_numbers, allow_punctuation):
        # Feature names
        self.input_feature = input_feature
        self.output_feature = output_feature

        # Cleaning parameters
        self.split_level = split_level
        self.case_sensitive = case_sensitive
        self.allow_numbers = allow_numbers
        self.allow_punctuation = allow_punctuation

    def clean(self, text):
        """Basic text preprocessing."""

        # Case sensitive
        if not self.case_sensitive:
            text = " ".join(token.lower() for token in text.split(" "))

        # Split into tokens
        if self.split_level == "word":
            text = " ".join(token for token in text.split(" "))
        elif self.split_level == "char":
            text = " ".join(token for token in text)

        # Clean newlines
        text = text.replace("\n", " ")

        # Regex
        regex_expression = r"[^a-zA-Z]+"
        if self.allow_numbers:
            regex_expression += r"[^0-9]+"
        if self.allow_punctuation:
            regex_expression += r"[^.,?!:;-$%&()[]#]+"
        text = re.sub(regex_expression, r" ", text)

        # Remove leading and trailing spaces
        text = text.strip()

        return text

    def clean_df(self, df):
        # Rename columns
        column_names = {self.input_feature: "X", self.output_feature: "y"}
        df = df.rename(columns=column_names)

        # Clean inputs
        df.X = df.X.apply(func=self.clean)

        wrap_text("Preprocessed data")
        ml_logger.info(df.head(5))
        return df

    @classmethod
    def load(cls, preprocessor_filepath):
        contents = load_json(preprocessor_filepath)
        return cls(**contents)

    def save(self, preprocessor_filepath):
        with open(preprocessor_filepath, "w") as fp:
            json.dump(self.__dict__, fp, indent=4)


