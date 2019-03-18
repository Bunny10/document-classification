import pytest

from document_classification.utils import clean_text

@pytest.mark.parametrize("text, expected", [
    ("Goku Mohandas", "goku mohandas"),
    ("G8oku Mohandas 443", "g oku mohandas"),
    ("Goku, Mohandas!", "goku mohandas"),
])
def test_clean_text(text, expected):
    assert clean_text(text) == expected