import logging
import unittest
from pathlib import Path
from unittest import TestCase

import h5py
import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from testfixtures import TempDirectory

from dsconcept.get_metrics import HierarchicalClassifier, StubBestEstimator

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class TestHierarchicalClassifier(TestCase):
    def setUp(self) -> None:
        self.d = TempDirectory()
        self.clf_loc = Path(self.d.path) / "stub.jbl"
        out_info = {'concept': 'something', 'best_estimator_': StubBestEstimator()}
        joblib.dump(out_info, self.clf_loc)
        cat_clfs = [
            {"best_estimator_": StubBestEstimator(), "concept": "physics"},
            {"best_estimator_": StubBestEstimator(), "concept": "video games"},
        ]
        kwd_clfs = {
            ("physics", "gamma ray"): StubBestEstimator(),
            ("video games", "minecraft"): StubBestEstimator(),
            ("video games", "kerbal space program"): StubBestEstimator(),
            ("", "minecraft"): StubBestEstimator(),
            ("", "gamma ray"): StubBestEstimator(),
            ("", "penguins"): StubBestEstimator(),
        }
        kwd_clfs_locs = {
            ("physics", "gamma ray"): self.clf_loc,
            ("video games", "minecraft"): self.clf_loc,
            ("video games", "kerbal space program"): self.clf_loc,
            ("", "minecraft"): self.clf_loc,
            ("", "gamma ray"): self.clf_loc,
            ("", "penguins"): self.clf_loc,
        }
        self.hclf = HierarchicalClassifier(cat_clfs, kwd_clfs)
        self.hclf_locs = HierarchicalClassifier(cat_clfs, kwd_clfs_locs)
        self.feature_matrix = np.array([[0, 0, 1, 1], [0, 1, 0, 1], [1, 1, 0, 0]])
        v = DictVectorizer()
        d = [{"astronauts": 1, "astronomy": 1}, {"space": 1, "basalt": 1}]
        v.fit(d)
        self.v = v

    def test_cat_clfs(self):
        cats = ["physics", "video games", ""]
        self.assertListEqual(self.hclf.categories, cats)

    def test_kwd_clfs(self):
        kwds = ["gamma ray", "kerbal space program", "minecraft", "penguins"]
        self.assertListEqual(self.hclf.concepts_with_classifiers.tolist(), kwds)

    def test_predict_categories(self):
        cat_preds = self.hclf.predict_categories(self.feature_matrix)
        self.assertEqual(cat_preds.shape, (3, 2))
        print(cat_preds)

    def test__predict_one_clf(self):
        pred = self.hclf._predict_one_clf(self.feature_matrix, 1, "video games")
        self.assertEqual(pred.shape[0], 3)

    def test__predict_one_clf_locs(self):
        pred = self.hclf_locs._predict_one_clf(self.feature_matrix, 1, "video games")
        self.assertEqual(pred.shape[0], 3)

    def test__predict_keywords(self):
        cat_indices = {"physics": [0], "video games": [1, 2]}
        store = self.hclf._predict_keywords(
            self.feature_matrix,
            f"{self.d.path}/store.h5",
            cat_indices,
            only_no_topic=False,
            use_dask=False,
        )
        with h5py.File(store, 'r') as f0:
            pred_array = f0["predictions"][()]
        LOG.info(pred_array)
        self.assertEqual(pred_array.shape, (3, 3, 4))

    def test__predict_keywords_locs(self):
        cat_indices = {"physics": [0], "video games": [1, 2]}
        store = self.hclf_locs._predict_keywords(
            self.feature_matrix,
            f"{self.d.path}/store.h5",
            cat_indices,
            only_no_topic=False,
            use_dask=False,
        )
        with h5py.File(store, 'r') as f0:
            pred_array = f0["predictions"][()]
        LOG.info(pred_array)
        self.assertEqual(pred_array.shape, (3, 3, 4))

    def test_get_synth_preds(self):
        cat_indices = {"physics": [0], "video games": [1, 2]}
        store = self.hclf._predict_keywords(
            self.feature_matrix,
            f"{self.d.path}/store.h5",
            cat_indices,
            only_no_topic=False,
            use_dask=False,
        )
        all_cat_inds = {
            "physics": np.array([0]),
            "video games": np.array([0, 1]),
            "": np.array([0, 1, 2]),
        }
        self.hclf.get_synth_preds(
            store,
            all_cat_inds,
            batch_size=10000,
            only_cat=False,
            synth_strat="mean",
            use_dask=False,
        )
        with h5py.File(store) as f0:
            synth_array = f0["synthesis"].value
        LOG.info(synth_array)
        self.assertEqual(synth_array.shape, (3, 4))

    def test__to_strings(self):
        synth_array = np.array(
            [[0, 0.51, 0.9, 0.2], [0.8, 0.1, 0.4, 0.7], [0.4, 0.2, 0.1, 0.9]]
        )
        kwd_strs = self.hclf._to_strings(
            self.hclf.concepts_with_classifiers, synth_array, t=0.5
        )
        results = [
            [("minecraft", 0.9), ("kerbal space program", 0.51)],
            [("gamma ray", 0.8), ("penguins", 0.7)],
            [("penguins", 0.9)],
        ]
        self.assertEqual(results, kwd_strs)
        LOG.info(kwd_strs)

    def test_predict(self):
        examples = [
            "Olympus Mons is the largest volcano in the solar system",
            "Database management is critical for information retrieval",
            "We used a logistic regression with batched stochastic gradient descent.",
        ]
        weights = {"NOUN": 1, "PROPN": 1, "ENT": 1, "NOUN_CHUNK": 1, "ACRONYM": 1}
        self.hclf.vectorizer = self.v
        features, feature_matrix = self.hclf.vectorize(examples, weights)
        self.hclf.predict(feature_matrix)


if __name__ == "__main__":
    unittest.main()