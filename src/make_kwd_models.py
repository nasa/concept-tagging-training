import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

import dsconcept.model as ml
import dsconcept.train as tr

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)

VECTORIZER = "vectorizer.jbl"
FEATURE_MATRIX = "feature_matrix.jbl"
OUT_MODELS_DIR = "models/topic_"


def main(
    in_feature_matrix,
    in_ind_train,
    in_ind_test,
    in_kwd_indices,
    in_cat_indices,
    in_kwd_raw2lemma,
    in_cat_raw2lemma,
    in_config,
    out_dir,
    topics=True,
):
    with open(in_config, "r") as f0:
        config = yaml.safe_load(f0)

    X = joblib.load(in_feature_matrix)
    ind_train = np.load(in_ind_train)
    ind_test = np.load(in_ind_test)

    LOG.info(f"Loading keyword extractor from {in_kwd_indices} and {in_kwd_raw2lemma}.")
    ce = ml.ConceptExtractor()
    ce.from_jsons(in_kwd_indices, in_kwd_raw2lemma)

    LOG.info(
        f"Loading category extractor from {in_cat_indices} and {in_cat_raw2lemma}."
    )
    cat_ext = ml.ConceptExtractor()
    cat_ext.from_jsons(in_cat_indices, in_cat_raw2lemma)

    paramgrid = {
        "alpha": [0.01, 0.001, 0.0001],
        "class_weight": [{1: 10, 0: 1}, {1: 5, 0: 1}, {1: 20, 0: 1}],
        "max_iter": [5],
        "loss": ["log"],
    }  # requires loss function with predict_proba
    clf = GridSearchCV(
        SGDClassifier(), paramgrid, scoring="f1", n_jobs=-1,
    )  # requires GridSearchCV
    out_models = out_dir / OUT_MODELS_DIR
    trainer = tr.ConceptTrainer(ce, clf)
    doc_topic_indices = cat_ext.concept_index_mapping

    if topics:
        LOG.info(
            f"Training one set for each of {len(doc_topic_indices)} topics divisions."
        )
        for topic, doc_topic_index in doc_topic_indices.items():
            trainer.train_concepts(
                X,
                ind_train,
                ind_test,
                out_models,
                config["min_concept_occurrence"],
                topic,
                doc_topic_index,
            )
    LOG.info("Training one general set")
    trainer.train_concepts(
        X, ind_train, ind_test, out_models, config["min_concept_occurrence"]
    )
    LOG.info("Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Use feature matrix and location of indices to create classifiers
        for the concepts in the corpus."""
    )
    parser.add_argument(
        "in_feature_matrix", help="input scipy sparse matrix of features"
    )
    parser.add_argument("in_ind_train", help="train set index")
    parser.add_argument("in_ind_test", help="test set index")
    parser.add_argument("in_kwd_indices", help="keyword indicies")
    parser.add_argument("in_cat_indices", help="category indices")
    parser.add_argument("in_kwd_raw2lemma", help="keyword raw to lemma mapping")
    parser.add_argument("in_cat_raw2lemma", help="category raw to lemma mapping")
    parser.add_argument("in_config", help="configuration for creating models")
    parser.add_argument(
        "out_dir",
        help="output directory for vectorizer, feature matrix, and models",
        type=Path,
    )
    parser.add_argument("--topics", dest="topics", action="store_true")
    parser.add_argument("--no-topics", dest="topics", action="store_false")
    parser.set_defaults(topics=True)
    parser.add_argument(
        "-v", "--verbose", help="increase output verbosity", action="store_true"
    )
    args = parser.parse_args()
    if args.verbose:
        LOG.info("Changing log level to DEBUG.")
        LOG.setLevel(logging.DEBUG)
        tr.LOG.setLevel(logging.DEBUG)
        LOG.debug("Changed log level to DEBUG.")

    main(
        args.in_feature_matrix,
        args.in_ind_train,
        args.in_ind_test,
        args.in_kwd_indices,
        args.in_cat_indices,
        args.in_kwd_raw2lemma,
        args.in_cat_raw2lemma,
        args.in_config,
        args.out_dir,
        args.topics,
    )
