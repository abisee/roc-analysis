import json
import os
from ROC import fixed_settings
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize

"""To set options"""
class Options(object):
    def __init__(self):
        return
    def __str__(self):
        return json.dumps(vars(self), indent=3)

"""To set options of ...
Default values are set to enable fast end to end training useful for debugging."""
class Defaults(Options):
    def __init__(self):
        # Data
        self.ROC_FILENAME = 'ROCStories_winter2017.csv'
        self.ROC_FILEPATH = os.path.join(fixed_settings.DATA_ROOT, self.ROC_FILENAME)
        self.NUM_STORIES = 10

        self.preprocess=True

        # TF-IDF options
        self.tfidf_norm = None
        self.tfidf_max_df=1
        self.tfidf_sublinear_tf = False
        self.tfidf_binary_tf = True

        # Ids to fetch neighbors for
        self.SAMPLES = [0, 1, 2, 3, 4, 5]
        # Number of neighbors to fetch
        self.NUM_NEIGHBORS = 3
        self.SIMILARITY_THRESHOLD = None