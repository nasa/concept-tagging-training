import logging
from pathlib import Path
from unittest import TestCase

import joblib
import numpy as np
from scipy.sparse import csc_matrix
from hypothesis import given
from hypothesis.extra.numpy import arrays
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from testfixtures import TempDirectory

from .context import dsconcept

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class TestConceptTrainer(TestCase):
    def setUp(self):
        ce = dsconcept.model.ConceptExtractor()
        fe = dsconcept.model.FeatureExtractor()
        self.d = TempDirectory()
        data = b'{"abstract":["Astronauts are very cool."], "concept": ["ASTRONAUTS", "COOL THINGS"]}\n {"abstract":["NASA is going to Mars."], "concept":["NASA", "MARS"]}'
        self.d.write("test.json", data)
        self.corpus_path = f"{self.d.path}/test.json"
        s = 100
        self.X = csc_matrix(np.random.randint(2, size=s * 2).reshape(int(s), 2))
        self.y = np.random.randint(2, size=s)
        paramgrid = {
            "alpha": [0.01, 0.001, 0.0001],
            "class_weight": [{1: 10, 0: 1}, {1: 5, 0: 1}, {1: 20, 0: 1}],
            "max_iter": [1],
            "loss": ["log"],
        }  # requires loss function with predict_proba
        clf = GridSearchCV(
            SGDClassifier(), paramgrid, scoring="f1"
        )  # requires GridSearchCV
        self.ct = dsconcept.train.ConceptTrainer(ce, clf)

    def test_create_concept_classifier(self):
        out_dir = Path(f"{self.d.path}/models")
        out_dir.mkdir()
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.5, random_state=42
        )
        self.ct.create_concept_classifier(
            "test_concept", X_train, X_test, y_train, y_test, out_dir
        )
        clf = joblib.load(out_dir / "test_concept.pkl")
        LOG.info(clf)

    def test_train_all(self):  # This test is super naive. Does not check behaviour.
        self.ct.train_all(self.X, Path(f"{self.d.path}/models"), 5)
        test_inds = np.load((Path(f"{self.d.path}") / "test_inds.npy"))
        train_inds = np.load((Path(f"{self.d.path}") / "train_inds.npy"))
        LOG.info(f"test_inds: {test_inds}")
        LOG.info(f"train_inds: {train_inds}")

    @given(arrays(dtype=np.float_, shape=1))
    def test_get_dispersed_subset(self, array):
        subset = dsconcept.train.get_dispersed_subset(array, 5)
        self.assertLessEqual(len(subset), 5)
        LOG.info(subset)

    def tearDown(self):
        self.d.cleanup()
