import os
from pathlib import Path

import nltk

# Use the NLTK data bundled with this project
NLTK_DATA_DIR = Path(__file__).resolve().parent / "nltk_data"

os.environ["NLTK_DATA"] = str(NLTK_DATA_DIR)

if str(NLTK_DATA_DIR) not in nltk.data.path:
    nltk.data.path.insert(0, str(NLTK_DATA_DIR))

# Verify that required NLTK resources are available
nltk.data.find("corpora/stopwords")
nltk.data.find("tokenizers/punkt_tab/english/")

print("NLTK resources loaded successfully.")