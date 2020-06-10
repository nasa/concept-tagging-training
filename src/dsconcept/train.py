"""
Train
-----
Program to make classifiers from input corpus and selected keyword field.

author: Anthony Buonomo
contact: anthony.r.buonomo@nasa.gov

"""
import logging
from pathlib import Path
import time
from math import ceil

from sklearn.model_selection import train_test_split
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)
import joblib
import numpy as np
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=Warning)
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def get_dispersed_subset(array, subset_size):
    """
    Get dispersed subset of an array. By dispersed, I mean that we values extract values
    from an evenly distributed by location in the array.

    Args:
        array (numpy.ndarray): array from which to extract subset
        subset_size (int): the number of elements to extract from array

    Returns:
        subset (numpy.ndarray): the dispersed subset
        array (numpy.ndarray): if subset_size too large, return the input array

    Examples:
        >>> from dsconcept.train import get_dispersed_subset
        >>> l1 = list(range(100))
        >>> l2 = get_dispersed_subset(l1, 10)
        >>> l2
        array([ 0., 12., 23., 34., 45., 56., 67., 78., 89., 99.], dtype=float16)
    """
    if len(array) <= subset_size:
        return array
    else:
        last = array[-1]
        subset = [sub[0] for sub in np.array_split(array, (subset_size - 1))]
        subset.append(last)
        subset = np.array(subset, dtype=np.float16)
        return subset


# TODO: refactor to remove need for this function
def path_append(in_path, addition):
    out_path = f"{in_path.parent}/{in_path.stem}{addition}{in_path.suffix}"
    return Path(out_path)


def topic_path_format(out_classifier_dir, topic):
    if topic is None:
        tmp_topic = ""
    else:
        tmp_topic = topic
    out_classifier_dir = path_append(out_classifier_dir, tmp_topic)  # appends to stem
    if not out_classifier_dir.exists():
        out_classifier_dir.mkdir(parents=True)
    return out_classifier_dir


class ConceptTrainer:
    def __init__(self, concept_extractor, classifier):
        """
        Initialize object for training of classifiers based on given corpus extractors.

        Args:
            concept_extractor (dsconcept.model.ConceptExtractor): ConceptExtractor (with concepts already loaded)
                for which to create classifiers
            classifier (sklearn.GridSearchCV): the classifier algorithm to use (wrapped in sklearn GridSearchCV)

        """
        self.concept_extractor = concept_extractor
        self.classifier = classifier

    def train_all(
        self,
        doc_feature_matrix,
        out_classifier_dir,
        min_concept_freq,
        doc_topic_distr=None,
    ):
        """
        Train classifiers for each concept for each topic (if topic distributions are provided).

        Args:
            doc_feature_matrix (scipy.sparse.csr.csr_matrix): document feature matrix
            out_classifier_dir (pathlib.Path): output path for classifiers
            min_concept_freq (int): minimum frequency for concepts in corpus in order
                for their corresponding classifiers to be made
            doc_topic_distr (numpy.ndarray): topic distributions for each doc in training set

        Returns:
            out_classifier_dir (pathlib.Path): output path for classifiers

        """
        doc_topic_indices = {}
        if doc_topic_distr is not None:
            for topic in range(
                doc_topic_distr.shape[1]
            ):  # cols of distr matrix ~ topics
                doc_topic_indices[topic] = [
                    i
                    for i, distr in enumerate(doc_topic_distr)
                    if distr.argmax() == topic
                ]
        _, _, ind_train, ind_test = train_test_split(
            doc_feature_matrix,
            np.array(range(doc_feature_matrix.shape[0])),
            test_size=0.10,
            random_state=42,
        )
        np.save(out_classifier_dir.parent / f"train_inds.npy", ind_train)
        np.save(out_classifier_dir.parent / f"test_inds.npy", ind_test)

        LOG.info(
            f"Training one general set, and one set for each of {len(doc_topic_indices)} topics divisions."
        )
        for topic, doc_topic_index in doc_topic_indices.items():
            self.train_concepts(
                doc_feature_matrix,
                ind_train,
                ind_test,
                out_classifier_dir,
                min_concept_freq,
                topic,
                doc_topic_index,
            )
        self.train_concepts(
            doc_feature_matrix,
            ind_train,
            ind_test,
            out_classifier_dir,
            min_concept_freq,
        )
        return out_classifier_dir

    def train_concepts(
        self,
        doc_feature_matrix,
        ind_train,
        ind_test,
        out_classifier_dir,
        min_concept_freq,
        topic=None,
        doc_topic_index=None,
        scale_threshold=False,
    ):
        """
        Create classifiers for group of concepts.

        Args:
            doc_feature_matrix (scipy.sparse.csr.csr_matrix): document feature matrix
            ind_train (list of int): indices for training partition
            ind_test (list of int): indices for testing partition
            out_classifier_dir (pathlib.Path): path to directory where classifiers will be dumped.
            min_concept_freq (int): minimum frequency for concepts in corpus in order
                for their corresponding classifiers to be made
            topic (int | None): the topic (if any) from which to select training data for classifiers
            doc_topic_index (lists): mapping from given topic to document indices
                for which that topic has the highest probability
            scale_threshold (bool | False): If true, scale the minimum_concept_freq by the size of the topic division.

        Returns:
            out_classifier_dir (pathlib.Path): directory where classifiers have been placed

        """

        LOG.info(f"Queuing classifier job for topic {topic}.")
        t1 = time.time()
        out_classifier_dir = topic_path_format(out_classifier_dir, topic)

        LOG.info("Getting indices for training and testing.")
        if doc_topic_index is not None:
            train_inds = list(set(ind_train).intersection(doc_topic_index))
            test_inds = list(set(ind_test).intersection(doc_topic_index))
        else:
            train_inds = ind_train
            test_inds = ind_test

        X_train = doc_feature_matrix.tocsc()[train_inds, :]
        X_test = doc_feature_matrix.tocsc()[test_inds, :]

        if scale_threshold:
            total_size = X_train.shape[0] + X_test.shape[0]
            # scale threshold based on size of topic division
            r = total_size / doc_feature_matrix.shape[0]
            topic_min_concept_threshold = ceil(min_concept_freq * r)
        else:
            topic_min_concept_threshold = min_concept_freq
        LOG.info(f"Topic threshold set to {topic_min_concept_threshold}.")

        concept_index_mapping = self.concept_extractor.get_top_concepts(
            topic_min_concept_threshold
        )
        no_concepts = len(concept_index_mapping)
        LOG.info(f"Training {no_concepts} concepts.")

        nu_passed = 0
        for concept, index in tqdm(concept_index_mapping.items()):
            LOG.debug(f"TOPIC={topic}:Loading indices for {concept}")
            y = np.zeros(doc_feature_matrix.shape[0])
            np.put(y, index, 1)

            y_train = y[train_inds]
            y_test = y[test_inds]
            total_yes = sum(y_train) + sum(y_test)

            if total_yes < topic_min_concept_threshold:
                nu_passed += 1
                LOG.debug(
                    f"Passing {concept} because it is under topic_min_concept_threshold of {topic_min_concept_threshold}."
                )
                continue
            # TODO: move around y0 train and test inds to keep aligned
            self.create_concept_classifier(
                concept, X_train, X_test, y_train, y_test, out_classifier_dir
            )
        t2 = time.time()
        LOG.warning(f"Passed {nu_passed} in topic {topic} due to freq under threshold.")
        LOG.debug(f"{t2-t1} seconds for topic {topic}.")
        return out_classifier_dir

    def create_concept_classifier(
        self, concept, X_train, X_test, y_train, y_test, out_classifier_dir
    ):
        """
        Create an individual classifier.

        Args:
            concept (str): the concept for which to create a classifier
            doc_feature_matrix (scipy.sparse.csr.csr_matrix): documents with their features
            y (numpy.ndarray): array which indicates whether or not given concept occurs for a given topic
            out_classifier_dir (pathlib.Path): output directory for classifiers

        Returns:
            out_model_path (pathlib.Path): the path to the concept classifier just produced.

        """
        LOG.debug(f"Making classifier for concept {concept}.")
        try:
            LOG.debug(f"fitting {concept}...")
            self.classifier.fit(X_train, y_train)
            LOG.debug(f"testing {concept}...")
            y_score = self.classifier.predict_proba(X_test)[:, 1]
            LOG.debug(f"Binarizing score for {concept}...")
            y_pred = np.where(y_score > 0.5, 1, 0)

            LOG.debug(f"Getting metric scores for {concept}...")
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_score)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            out_model = {
                "concept": concept,
                "best_estimator_": self.classifier.best_estimator_,
                "cv_results_": self.classifier.cv_results_,
                "scores": {
                    "accuracy": accuracy,
                    "roc_auc": roc_auc,
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                },
            }
            LOG.debug(f"Accuracy: {accuracy} | ROC-AUC: {roc_auc} | F1: {f1}")
            out_concept = str(Path(concept).name)
            out_model_path = out_classifier_dir / f"{out_concept}.pkl"
            LOG.debug(f"Writing model to  {out_model_path}.")
            joblib.dump(out_model, out_model_path)
            return out_model_path

        except ValueError:
            LOG.debug(f"Insufficient data for concept {concept}.")
