import argparse
import logging
import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import yaml
from pathlib import Path
import numpy as np

import dsconcept.model as ml
from dsconcept.train import ConceptTrainer

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

VECTORIZER = "vectorizer.jbl"
FEATURE_MATRIX = "feature_matrix.jbl"
OUT_MODELS_DIR = "models"


def main(
    in_feature_matrix,
    in_ind_train,
    in_ind_test,
    in_cat_indices,
    in_cat_raw2lemma,
    in_config,
    out_dir,
):
    with open(in_config, "r") as f0:
        config = yaml.safe_load(f0)

    X = joblib.load(in_feature_matrix)
    ind_train = np.load(in_ind_train)
    ind_test = np.load(in_ind_test)

    LOG.info(
        f"Loading category extractor from {in_cat_indices} and {in_cat_raw2lemma}."
    )
    cat_ext = ml.ConceptExtractor()
    cat_ext.from_jsons(in_cat_indices, in_cat_raw2lemma)

    paramgrid = {
        "alpha": [0.01, 0.001, 0.0001],
        "class_weight": [{1: 10, 0: 1}, {1: 5, 0: 1}, {1: 20, 0: 1}],
        "max_iter": [1],
        "loss": ["log"],
    }  # requires loss function with predict_proba
    clf = GridSearchCV(
        SGDClassifier(), paramgrid, scoring="f1"
    )  # requires GridSearchCV
    out_models = out_dir / OUT_MODELS_DIR
    trainer = ConceptTrainer(cat_ext, clf)

    trainer.train_concepts(
        X, ind_train, ind_test, out_models, config["min_concept_occurrence"]
    )
    LOG.info("Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Use feature matrix and location of indices to create classifiers
        for the categories in the corpus."""
    )
    parser.add_argument(
        "in_feature_matrix", help="input scipy sparse matrix of features"
    )
    parser.add_argument("in_ind_train", help="train set index")
    parser.add_argument("in_ind_test", help="test set index")
    parser.add_argument("in_cat_indices", help="category indices")
    parser.add_argument("in_cat_raw2lemma", help="category raw to lemma mapping")
    parser.add_argument("in_config", help="configuration for creating models")
    parser.add_argument(
        "out_dir",
        help="output directory for vectorizer, feature matrix, and models",
        type=Path,
    )
    args = parser.parse_args()

    main(
        args.in_feature_matrix,
        args.in_ind_train,
        args.in_ind_test,
        args.in_cat_indices,
        args.in_cat_raw2lemma,
        args.in_config,
        args.out_dir,
    )
