"""
Pipeline
--------
Program to make classifiers from input corpus and selected keyword field.

Author: Anthony Buonomo
Contact: anthony.r.buonomo@nasa.gov

Full opinionated pipeline from processing, to topic_modeling, to training classifiers.
"""

import logging
import warnings
from pathlib import Path

import plac
import yaml
from sklearn.decomposition import LatentDirichletAllocation
import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

import dsconcept.model as ml
from dsconcept.train import ConceptTrainer

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

FEATURES = Path("features.jsonl")
INDICES = Path("indices.json")
RAW2LEMMA = Path("raw2lemma.json")

TOPIC_VECTORIZER = Path("vectorizer.pkl")
TOPIC_FEATURE_MATRIX = Path("doc_feature_matrix.pkl")
TOPIC_MODEL = Path("model.pkl")
DOC_TOPIC_DISTR = Path("doc_topic_distr.pkl")

VECTORIZER = Path("vectorizer.pkl")
FEATURE_MATRIX = Path("doc_feature_matrix.pkl")
OUT_MODELS_DIR = Path("classifiers")


def process(
    in_corpus, out_dir, abstract_field, concept_field, term_types, batch_size, n_threads
):
    out_dir.mkdir(exist_ok=True, parents=True)
    out_features = out_dir / FEATURES
    out_indices = out_dir / INDICES
    out_raw2lemma = out_dir / RAW2LEMMA

    fe = ml.FeatureExtractor()
    fe.from_corpus_to_jsonlines(
        in_corpus, out_features, abstract_field, term_types, batch_size, n_threads
    )

    ce = ml.ConceptExtractor()
    ce.from_corpus(in_corpus, concept_field)
    ce.to_jsons(out_indices, out_raw2lemma)

    return fe, ce


def topic_model(
    topic_model_dir, processed_dir, topic_weights, min_feature, max_feature
):
    topic_model_dir.mkdir(exist_ok=True)
    tfe = ml.FeatureExtractor()
    tfe.from_jsonlines(processed_dir / FEATURES)

    topic_weighted_features = tfe.weight_terms(topic_weights)
    topic_limited_features = tfe.limit_features(
        topic_weighted_features, min_feature, max_feature
    )

    topic_v = DictVectorizer()
    topic_X = topic_v.fit_transform(topic_limited_features)

    model = LatentDirichletAllocation(
        n_components=3,
        max_iter=5,
        learning_method="online",
        learning_offset=50.0,
        random_state=0,
    )
    doc_topic_distr = model.fit_transform(topic_X)

    out_vectorizer = topic_model_dir / TOPIC_VECTORIZER
    out_feature_matrix = topic_model_dir / TOPIC_FEATURE_MATRIX
    out_model = topic_model_dir / TOPIC_MODEL
    out_doc_topic_distr = topic_model_dir / DOC_TOPIC_DISTR

    joblib.dump(topic_v, out_vectorizer)
    joblib.dump(topic_X, out_feature_matrix)
    joblib.dump(model, out_model)
    joblib.dump(doc_topic_distr, out_doc_topic_distr)

    return doc_topic_distr


def train(
    out_dir,
    process_dir,
    fe,
    ce,
    weights,
    min_feature,
    max_feature,
    min_concept_occurrence,
    doc_topic_distr,
):
    out_dir.mkdir(exist_ok=True)
    out_features = process_dir / FEATURES
    fe.from_jsonlines(out_features)
    weighted_features = fe.weight_terms(weights)
    limited_features = fe.limit_features(weighted_features, min_feature, max_feature)
    v = DictVectorizer()
    X = v.fit_transform(limited_features)

    out_vectorizer = out_dir / VECTORIZER
    out_feature_matrix = out_dir / FEATURE_MATRIX
    joblib.dump(v, out_vectorizer)
    joblib.dump(X, out_feature_matrix)

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
    trainer = ConceptTrainer(fe, ce, clf, out_models)
    trainer.train_all(
        X, out_models, min_concept_occurrence, doc_topic_distr=doc_topic_distr
    )
    return out_models


def parse_config(in_config):
    with open(in_config, "r") as f0:
        cfg = yaml.safe_load(f0)

    term_types = cfg["process"]["term_types"]
    abstract_field = cfg["process"]["abstract_field"]
    concept_field = cfg["process"]["concept_field"]

    topic_weights = cfg["topic_model"]["weights"]
    topic_min_feature = cfg["topic_model"]["min_feature_occurrence"]
    topic_max_feature = cfg["topic_model"]["max_feature_occurrence"]
    num_topics = cfg["topic_model"]["number_of_topics"]

    weights = cfg["train_classifiers"]["weights"]
    min_feature = cfg["train_classifiers"]["max_feature_occurrence"]
    max_feature = cfg["train_classifiers"]["max_feature_occurrence"]
    min_concept = cfg["train_classifiers"]["min_concept_occurrence"]

    return (
        abstract_field,
        concept_field,
        term_types,
        topic_weights,
        topic_min_feature,
        topic_max_feature,
        num_topics,
        weights,
        min_feature,
        max_feature,
        min_concept,
    )


@plac.annotations(
    in_corpus=plac.Annotation("path to json-formatted corpus", "positional", type=Path),
    config=plac.Annotation("path to configuration yaml file", "positional", type=Path),
    process_dir=plac.Annotation(
        "path to dir where you want to store processed corpus data",
        "positional",
        type=Path,
    ),
    topic_model_dir=plac.Annotation(
        "path to dir where you want to store topic_modeling data",
        "positional",
        type=Path,
    ),
    classify_dir=plac.Annotation(
        "path to dir where you want to store classifying data", "positional", type=Path
    ),
    batch_size=plac.Annotation(
        "size of batches to process in processing phase of pipeline", "option", type=int
    ),
    n_threads=plac.Annotation(
        "number of threads to use in processing phase of pipeline", "option", type=int
    ),
)
def main(
    in_corpus,
    config,
    process_dir,
    topic_model_dir,
    classify_dir,
    batch_size=10,
    n_threads=1,
):

    (
        abstract_field,
        concept_field,
        term_types,
        topic_weights,
        topic_min_feature,
        topic_max_feature,
        num_topics,
        weights,
        min_feature,
        max_feature,
        min_concept,
    ) = parse_config(config)

    fe, ce = process(
        in_corpus,
        process_dir,
        abstract_field,
        concept_field,
        term_types,
        batch_size,
        n_threads,
    )
    doc_topic_distr = topic_model(
        topic_model_dir,
        process_dir,
        topic_weights,
        topic_min_feature,
        topic_max_feature,
    )
    train(
        classify_dir,
        process_dir,
        fe,
        ce,
        weights,
        min_feature,
        max_feature,
        min_concept,
        doc_topic_distr,
    )
    LOG.info("SUCCESS!")


if __name__ == "__main__":
    plac.call(main)
