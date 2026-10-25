
import os
from pathlib import Path

import nltk

NLTK_DATA_DIR = Path(__file__).resolve().parent / "nltk_data"

# Make the directory available to NLTK and libraries that read
# the NLTK_DATA environment variable.
os.environ["NLTK_DATA"] = str(NLTK_DATA_DIR)

if str(NLTK_DATA_DIR) not in nltk.data.path:
    nltk.data.path.insert(0, str(NLTK_DATA_DIR))

# Fail early if the corpus isn't included in the deployment.
nltk.data.find("corpora/stopwords")
