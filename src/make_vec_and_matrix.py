import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

import dsconcept.model as ml

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

VECTORIZER = "vectorizer.jbl"
FEATURE_MATRIX = "feature_matrix.jbl"


def main(in_features, in_config, out_feature_dir, out_vectorizer):
    with open(in_config, "r") as f0:
        config = yaml.safe_load(f0)

    LOG.info(f"Loading features from {in_features}.")
    fe = ml.FeatureExtractor()
    fe.from_jsonlines(in_features)
    weighted_features = fe.weight_terms(config["weights"])
    limited_features = fe.limit_features(
        weighted_features,
        config["min_feature_occurrence"],
        config["max_feature_occurrence"],
    )
    v = DictVectorizer()
    X = v.fit_transform(limited_features)

    out_feature_matrix = out_feature_dir / FEATURE_MATRIX
    LOG.info(f"Outputting vectorizer to {out_vectorizer}.")
    joblib.dump(v, out_vectorizer)
    LOG.info(f"Outputting feature matrix to {out_feature_matrix}.")
    joblib.dump(X, out_feature_matrix)

    _, _, ind_train, ind_test = train_test_split(
        X, np.array(range(X.shape[0])), test_size=0.10, random_state=42
    )
    np.save(out_feature_dir / f"train_inds.npy", ind_train)
    np.save(out_feature_dir / f"test_inds.npy", ind_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""From features file, create a feature matrix and vectorizer 
        which translates between columns of the matrix and feature strings. Limit
        which features are included in these files with configuration."""
    )
    parser.add_argument("in_features", help="input features jsonlines file")
    parser.add_argument("in_config", help="configuration for creating models")
    parser.add_argument(
        "out_feature_dir",
        help="output directory for feature matrix and indices",
        type=Path,
    )
    parser.add_argument(
        "out_vectorizer", help="output path for for vectorizer", type=Path,
    )
    # TODO: split outputs
    args = parser.parse_args()

    main(args.in_features, args.in_config, args.out_feature_dir, args.out_vectorizer)
