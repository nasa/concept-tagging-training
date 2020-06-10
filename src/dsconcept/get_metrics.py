import argparse
import os
import logging
from math import ceil
from multiprocessing import cpu_count
from pathlib import Path
from typing import List, Dict, Tuple

import dask
import h5py
import joblib
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    recall_score,
    precision_score,
)
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm as tqdm
from tempfile import NamedTemporaryFile, TemporaryDirectory

import dsconcept.model as ml

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

PRED_LIST_TYPE = List[List[Tuple[str, float]]]


def get_cat_inds(
    categories: List[str], cat_preds: np.array, t: float = 0.5
) -> Dict[str, np.array]:
    """
    Apply a threshold to get documents indices corresponding to each category.

    Args:
        categories: list of categories which are columns of the cat_preds array
        cat_preds: array of scores for each category for each document
            ([documents, categories])
        t: threshold over which a category is determined to be relevant
            to a given document

    Returns:
        all_cat_inds: dictionary with keys which are categories.
            Values are index of documents which apply to each category.

    Examples:
        >>> from get_metrics import get_cat_inds
        >>> import numpy as np
        >>> cats = ['physics', 'geology']
        >>> cat_preds = np.array([[0.4, 0.8], [0.5, 0.6], [0.9, 0.3]])
        >>> get_cat_inds(cats, cat_preds, t=0.5)
        {'physics': array([2]), 'geology': array([0, 1])}
    """
    all_cat_inds = {}
    for i, cat in enumerate(categories):
        if cat == "":
            continue
        x = cat_preds[:, i]
        g_args = np.argwhere(x > t)
        if g_args.shape[0] == 0:
            cat_inds = np.array([])
        else:
            cat_inds = np.stack(np.argwhere(x > t), axis=1)[0]
        all_cat_inds[cat] = cat_inds
    return all_cat_inds


def f_score(r: float, p: float, b: int = 1):
    """
    Calculate f-measure from recall and precision.

    Args:
        r: recall score
        p: precision score
        b: weight of precision in harmonic mean

    Returns:
        val: value of f-measure
    """
    try:
        val = (1 + b ** 2) * (p * r) / (b ** 2 * p + r)
    except ZeroDivisionError:
        val = 0
    return val


def get_mets(
    i: int,
    synth_preds: np.array,
    target_vals: np.array,
    con_with_clf: np.array,
    pbar=None,
) -> dict:
    """
    Get various metrics for the given arrays.
    #
    TODO: just pass in the already sliced synth_preds, Y, and con_with_clf?

    Args:
        i: index for the given concept
        synth_preds: arrays of predictions for each document and each concept
        target_vals: true values for each document and concept
        con_with_clf: arrays of concepts corresponding
            to columns synth_preds and target_vals

    Returns:
        metrics: metric records for the given concept
    """
    tmp_y_pred = synth_preds[:, i]
    tmp_y_pred_bool = [1 if v > 0.5 else 0 for v in tmp_y_pred]
    tmp_y_test = target_vals[:, i]
    p = precision_score(tmp_y_test, tmp_y_pred_bool)
    r = recall_score(tmp_y_test, tmp_y_pred_bool)
    f = f_score(r, p)
    accuracy = accuracy_score(tmp_y_test, tmp_y_pred_bool)
    try:
        roc_auc = roc_auc_score(tmp_y_test, tmp_y_pred)
    except ValueError:  # why does this happen?
        roc_auc = np.nan
    metrics = {
        "concept": con_with_clf[i],
        "accuracy": accuracy,
        "f1": f,
        "precision": p,
        "recall": r,
        "roc_auc": roc_auc,
    }
    if pbar is not None:
        pbar.update(1)
    return metrics


def synth_mean(
    kwd_preds_tmp: np.array, doc_index: int, concept_index: int, non_zero_cats: list,
) -> float:
    """
    Get the mean of nonzero predictions for given concept and given document.
    # TODO: get the precise matrix outside of function? Then pass in?

    Args:
        kwd_preds_tmp: 3D array of predictions
            [categories, documents, concepts]
        doc_index: index of test document
        concept_index: index of concept
        non_zero_cats: categories for which this concept has nonzero prediction

    Returns:
        mean: mean of nonzero predictions for this concept for this document
    """
    if len(non_zero_cats) != 0:
        mean = np.mean(kwd_preds_tmp[non_zero_cats, doc_index, concept_index])
    else:
        mean = np.nan
    return mean


def synth_max(
    kwd_preds_tmp: np.array, doc_index: int, concept_index: int, non_zero_cats: list,
) -> float:
    """
    Get the max of nonzero predictions for given concept and given document.
    # TODO: nearly same as above function. Just pass in the np.nanmax or mean as args and collapse into one function?
    """
    if len(non_zero_cats) != 0:
        val = np.nanmax(kwd_preds_tmp[non_zero_cats, doc_index, concept_index])
    else:
        val = np.nan
    return val


def get_means_for_one_doc(
    doc_index: int,
    all_cat_inds: Dict[str, np.array],
    kwd_preds_tmp: np.array,
    categories: List[str],
    no_cat_ind: int,
    only_cat: bool = False,
    synth_strat: str = "mean",
    pbar=None,
) -> np.array:
    """
    Get mean of nonzero concept predictions for each concepts
        in relevant categories for given doc.

    Args:
        doc_index: index of given document
        all_cat_inds: dictionary with keys which are categories.
            Values are index of documents which apply to each category.
        kwd_preds_tmp: array of all predictions
            [categories, documents, concepts]
        categories: list of categories
        no_cat_ind: index in categories list of the blank category ""
        only_cat: Only use category classifier or mixin the no category classifiers
        synth_strat: either "mean" or "max"
        # TODO: just pass a function instead of string?

    Returns:
        kwd_vals: array of synthesizes keyword prediction values
            for given document
    """
    cats = [
        cat for cat, inds in all_cat_inds.items() if doc_index in inds
    ]  # get category by index instead? means all_cat index should be by index
    cat_inds = [categories.index(cat) for cat in cats]
    if only_cat is False:
        cat_inds.append(no_cat_ind)
    # ^ also average with the no-topic set, make this a decision?
    kwd_vals = []
    for concept_index in range(kwd_preds_tmp.shape[2]):
        non_zero_cats = np.where(kwd_preds_tmp[:, doc_index, concept_index] != 0)[0]
        non_zero_cats = list(set(non_zero_cats).intersection(set(cat_inds)))
        assert synth_strat in ["mean", "max"], LOG.exception(
            f'Synthesis strategy "{synth_strat}" is invalid.'
        )
        strat = synth_mean if synth_strat == "mean" else synth_max
        v = strat(kwd_preds_tmp, doc_index, concept_index, non_zero_cats)
        kwd_vals.append(v)
    kwd_vals = np.array(kwd_vals)
    if pbar is not None:
        pbar.update(1)
    return kwd_vals


def create_ground_truth(
    store: str,
    dataset: str,
    test_inds: np.array,
    train_inds: np.array,
    concepts_with_classifiers: np.array,
    kwd_ext: ml.ConceptExtractor,
    batch_size: int,
):
    """
    Make an array of ground truth binary labels.

    Args:
        store: location of h5 database
        dataset: name of dataset in h5 database
            at which to store ground_truth array
        test_inds: test indices in the training data
        train_inds: training indices in the training data
        concepts_with_classifiers: all concepts which have models
        kwd_ext: ml.ConceptExtractor with ground_truth indices for concepts
        batch_size: batch_size for creating ground truth for each concept

    Returns:
        store, dataset: h5 store location and dataset name
    """
    with h5py.File(store, "a") as f0:
        ground_truth = f0.create_dataset(
            dataset,
            shape=(len(test_inds), len(concepts_with_classifiers)),
            compression="gzip",
        )
        n_batches = np.int(np.ceil(len(concepts_with_classifiers) / batch_size))
        for n in tqdm(range(n_batches)):
            start_batch = n * batch_size
            end_batch = (n + 1) * batch_size
            if end_batch >= len(concepts_with_classifiers):
                end_batch = len(concepts_with_classifiers) - 1
            batch_matrix = np.zeros((len(test_inds), end_batch - start_batch))
            con_batch = concepts_with_classifiers[start_batch:end_batch]
            for i, con in enumerate(con_batch):
                index = kwd_ext.concept_index_mapping[con]
                y_full = np.zeros((len(test_inds) + len(train_inds)))
                y_full[index] = 1
                y = y_full[test_inds]
                batch_matrix[:, i] = y
            ground_truth[:, start_batch:end_batch] = batch_matrix

    return store, dataset


# TODO: maybe make this a part of the hierarchical class
def get_synth_preds(
    store,
    shape,
    all_cat_inds,
    categories,
    batch_size,
    only_cat,
    synth_strat,
    use_dask=True,
    con_limit=None,
    limit=None,
    pbar=None,
):
    with h5py.File(store, "a") as f_synth, h5py.File(store, "r") as f_preds:
        if "synthesis" in f_synth.keys():
            del f_synth['synthesis']
        f_synth.create_dataset("synthesis", shape)
        synth_preds = f_synth["synthesis"]
        if (limit is not None):
            kwd_preds = f_preds["predictions"][:, 0:limit, :]
        else:
            kwd_preds = f_preds["predictions"]
        n_batches = np.ceil(kwd_preds.shape[1] / batch_size)
        LOG.debug(f"{n_batches} batches")
        no_cat_ind = categories.index("")
        for n in range(int(n_batches)):
            start_batch = n * batch_size
            end_batch = (n + 1) * batch_size
            if con_limit is not None:
                kwd_preds_tmp = kwd_preds[0:con_limit, start_batch:end_batch, :]
            else:
                kwd_preds_tmp = kwd_preds[:, start_batch:end_batch, :]
            n_docs = kwd_preds_tmp.shape[1]
            if True:  # use_dask is True:
                kwd_preds_tmp = dask.delayed(kwd_preds_tmp)
                all_cat_inds = dask.delayed(all_cat_inds)
                jobs = []
                for doc_index in range(n_docs):
                    # should be everything now, since '' category is included
                    job = dask.delayed(get_means_for_one_doc)(
                        doc_index,
                        all_cat_inds,
                        kwd_preds_tmp,
                        categories,
                        no_cat_ind,
                        synth_strat,
                        pbar=pbar,
                    )
                    jobs.append(job)
                hybrid_preds = dask.compute(jobs)[0]
            else:
                hybrid_preds = []
                for doc_index in range(n_docs):
                    # should be everything now, since '' category is included
                    v = get_means_for_one_doc(
                        doc_index,
                        all_cat_inds,
                        kwd_preds_tmp,
                        categories,
                        no_cat_ind,
                        only_cat,
                        synth_strat,
                        pbar=pbar,
                    )
                    hybrid_preds.append(v)
            hybrid_pred_array = np.stack(hybrid_preds)
            if limit is not None:
                if limit <= end_batch:
                    synth_preds[start_batch:limit, :] = hybrid_pred_array
                else:
                    synth_preds[start_batch:end_batch, :] = hybrid_pred_array
            else:
                synth_preds[start_batch:end_batch, :] = hybrid_pred_array


def load_category_models(in_cat_models: str) -> List[dict]:
    """
    Load all category models from given directory

    Args:
        in_cat_models: directory where category models reside

    Returns:
        cat_clfs: A list of dictionaries, each with a category model
    """
    LOG.info(f"Loading category classifiers from {in_cat_models}.")
    in_clfs = list(Path(in_cat_models).iterdir())
    cat_clfs = [joblib.load(c) for c in tqdm(in_clfs)]
    return cat_clfs


def load_concept_models(in_kwd_models: str, load: bool = True) -> Dict[Tuple[str, str], GridSearchCV]:
    """
    Load keyword models from given directory.

    Args:
        in_kwd_models: directory with subdirs, the suffixes of which are the
            names of the categories (ex. topic_physics). Each of these
            subfolders contains binary files for concepts in that category.
            The classifiers trained on all documents are in a subfolder which
            has not suffix (ex. topic_).
        load: whether to load the models into memory, or just get their paths

    Returns:
        cd: Dictionary with all classifiers for each category.
    """
    LOG.info(f"Loading keyword classifiers from {in_kwd_models}.")
    cd = {}  # expects no_topics with suffix ''
    topic_dirs = list(Path(in_kwd_models).iterdir())
    total = 0
    for td in topic_dirs:
        in_clfs = list(td.iterdir())
        total += len(in_clfs)
    pbar = tqdm(topic_dirs, total=total)
    for topic_dir in pbar:
        topic_name = topic_dir.stem.split("_")[1]  # depends on opinionated path format
        pbar.set_description(topic_name)
        in_clfs = list(topic_dir.iterdir())
        clfs = (joblib.load(c) for c in in_clfs)  # generator for loading classifiers
        for c, c_loc in zip(clfs, in_clfs):
            if load is True:
                cd[topic_name, c["concept"]] = c["best_estimator_"]
            else:
                cd[topic_name, c['concept']] = c_loc
            pbar.update(1)
    return cd


def make_predictions(
    in_cat_models,
    in_kwd_models,
    feature_matrix,
    out_store="test_results/store.h5",
    t=None,
):
    cat_clfs = load_category_models(in_cat_models)
    cd = load_concept_models(in_kwd_models)
    clf = HierarchicalClassifier(cat_clfs, cd)
    LOG.info("Predicting categories.")
    cat_preds = clf.predict_categories(feature_matrix)
    if t is not None:
        LOG.info("Only making predictions for keywords in predicted categories.")
        cat_indices = get_cat_inds(clf.categories, cat_preds, t)
        # TODO: add rule for when cat_indices has nothing in it!
        all_kwd_preds_loc = clf._predict_keywords(
            feature_matrix, out_store, cat_indices
        )
    else:
        LOG.info("Predicting for all keywords on all documents.")
        # TODO: this should call a public function
        all_kwd_preds_loc = clf._predict_keywords(feature_matrix, out_store)
    LOG.info(f"all_kwd_preds_loc={all_kwd_preds_loc}")

    return clf.categories, clf.concepts_with_classifiers, cat_preds


class StubBestEstimator:
    """
    Stub class for classifier's best_estimator to be used for testing.
    """

    def init(self):
        pass

    def predict_proba(self, feature_matrix):
        val = np.random.rand(feature_matrix.shape[0], 2)
        return val


def main(
    experiment_name, out_store, out_cat_preds, gt_batch_size, limit=None,
):
    LOG.info("Loading test data and models.")
    # TODO: paths should be put into main function
    test_inds = np.load(f"data/interim/{experiment_name}/test_inds.npy")
    train_inds = np.load(f"data/interim/{experiment_name}/train_inds.npy")
    feature_matrix = joblib.load(f"data/interim/{experiment_name}/feature_matrix.jbl")
    in_cat_models = Path(f"models/{experiment_name}/categories/models/")
    in_kwd_models = Path(f"models/{experiment_name}/keywords/models/")

    if limit is not None:
        LOG.info(f"Limiting to {limit} test records.")
        feature_matrix_test = feature_matrix.tocsc()[test_inds[0:limit], :]
        # TODO: How does this affect indices?
    else:
        feature_matrix_test = feature_matrix.tocsc()[test_inds, :]

    LOG.info("Making predictions.")
    categories, concepts_with_classifiers, cat_preds, = make_predictions(
        in_cat_models, in_kwd_models, feature_matrix_test, out_store,
    )  # need t if limiting
    np.save(out_cat_preds, cat_preds)
    LOG.info("Creating ground truth data.")
    kwd_ext = ml.ConceptExtractor()  # TODO: these paths should be provided as args
    kwd_ext.from_jsons(
        f"data/interim/{experiment_name}/kwd_indices.json",
        f"models/{experiment_name}/kwd_raw2lemma.json",
    )
    create_ground_truth(
        store=out_store,
        dataset="ground_truth",
        kwd_ext=kwd_ext,
        concepts_with_classifiers=concepts_with_classifiers,
        batch_size=gt_batch_size,
        train_inds=train_inds,
        test_inds=test_inds,
    )


def get_category_results(cat_models_dir: Path) -> pd.DataFrame:
    in_clfs = list(cat_models_dir.iterdir())
    cat_clfs = [joblib.load(c) for c in in_clfs]  # loads the classifiers
    cat_results_df = pd.DataFrame(
        [{**c["scores"], **{"concept": c["concept"]}} for c in cat_clfs]
    )
    return cat_results_df


def get_keyword_results(kwd_models_dir: Path) -> pd.DataFrame:
    cd = {}
    for topic_dir in kwd_models_dir.iterdir():
        in_clfs = list(topic_dir.iterdir())
        clfs = (joblib.load(c) for c in in_clfs)  # loads the classifiers
        topic_name = topic_dir.stem.split("_")[1]  # depends on opinionated path format
        cd[topic_name] = clfs

    all_records = []
    for t, clfs in tqdm(cd.items()):
        for clf in clfs:
            r = {**{"concept": clf["concept"], "category": t}, **clf["scores"]}
            all_records.append(r)
    results_df = pd.DataFrame(all_records)
    return results_df


class HierarchicalClassifier:
    """
    Hierarchical Classifier object which allows for streamlined predictions
        on suites of concept models associated with different categories.

    Attributes:
        categories: list of categories
        concepts_with_classifiers: sorted array of concepts with classifiers
        cat_concept_indices: list where each element maps onto a category.
            Each element consists of a selection of indices
            in concepts_with_classifier which occur in the given category.
        vectorizer: DictVectorizer for transforming features
    """

    def __init__(
        self, cat_clfs: List[dict], kwd_clfs: Dict[Tuple[str, str], GridSearchCV],
    ):
        """
        Set the models for categories and concepts_with_classifiers

        Args:
            cat_clfs: category classifier models
            kwd_clfs: Dictionary with keys which are tuples
                of categories and concepts, values are the classifier models
        """
        self.cat_clfs = cat_clfs
        self.kwd_clfs = kwd_clfs
        self.vectorizer = None

    @property
    def cat_clfs(self):
        """
        The category classifiers.

        Setter also creates categories attribute.
        """
        return self._cat_clfs

    @property
    def kwd_clfs(self):
        """
        Dictionary with keys which are tuples of categories and concepts,
            values are the classifier models

        Setter method creates concept_indices,
            and concepts_with_classifiers attributes.
        """
        return self._kwd_clfs

    @cat_clfs.setter
    def cat_clfs(self, cat_clfs: List[dict]):
        self._cat_clfs = cat_clfs
        self.categories = [c["concept"] for c in self.cat_clfs] + [""]

    @kwd_clfs.setter
    def kwd_clfs(self, kwd_clfs: Dict[Tuple[str, str], dict]):
        self._kwd_clfs = kwd_clfs
        category_concepts = {}

        for cat in self.categories:
            concepts = [k[1] for k, v in kwd_clfs.items() if k[0] == cat]
            # concepts = [clf["concept"] for clf in kwd_clfs[cat]]
            category_concepts[cat] = concepts

        all_cat_concepts = set(
            c for ts, cons in category_concepts.items() for c in cons
        )
        concepts_with_classifiers = np.sort(list(all_cat_concepts))
        LOG.info(f"concepts_with_classifiers: {concepts_with_classifiers.shape[0]}")

        cat_concept_indices = []
        for cat in self.categories:
            full_in_cats = np.isin(concepts_with_classifiers, category_concepts[cat])
            cat_concept_cols = np.where(full_in_cats)[0]
            cat_concept_indices.append(cat_concept_cols)

        self.cat_concept_indices: List[np.array] = cat_concept_indices
        # shape is [categories, keywords]
        self.concepts_with_classifiers: np.array = concepts_with_classifiers

    def load_vectorizer(self, v_loc: str):
        """
        Loads the DictVectorizer

        Args:
            v_loc: location of vectorizer
        """
        self.vectorizer: DictVectorizer = joblib.load(v_loc)

    def vectorize(
        self,
        texts: List[str],
        weights: Dict[str, int],
        batch_size: int = 1000,
        n_threads: int = cpu_count(),
    ) -> Tuple[List[Dict[str, str]], np.array]:
        """
        Transform texts into a matrix of features.

        Args:
            texts: texts to transform
            weights: how to weight different types of features
            batch_size: what batch size to pass to nlp.pipe
            n_threads: number of threads to use

        Returns:
            feature_matrix: matrix representation of features for each document
        """
        assert self.vectorizer is not None, LOG.exception("Must initialize vectorizer.")
        fe = ml.FeatureExtractor()
        with NamedTemporaryFile() as tmp_features_loc:
            tmp_features = tmp_features_loc.name
            ml.extract_features_from_abstracts(
                texts, tmp_features, batch_size, n_threads
            )
            fe.from_jsonlines(tmp_features)
        weighted_features = fe.weight_terms(weights)
        feature_matrix = self.vectorizer.transform(weighted_features)
        return fe.features, feature_matrix

    def predict_categories(self, feature_matrix: np.array) -> np.array:
        """
        Make predictions with category classifiers

        Args:
            feature_matrix: array of features for each document

        Returns:
            cat_preds: prediction belief values for each document
        """
        cat_preds_list = [
            clf["best_estimator_"].predict_proba(feature_matrix)[:, 1]
            for clf in tqdm(self.cat_clfs)
        ]
        cat_preds = np.stack(cat_preds_list, axis=1)
        return cat_preds

    def _predict_one_clf(
        self, feature_matrix: np.array, concept_index: int, cat: str, pbar=None,
    ) -> np.array:
        """
        Make a prediction for a particular concept.

        Args:
            feature_matrix: array of features for each document
            concept_index: index for the given concept
                in concepts_with_classifiers attribute
            cat: name of the given category

        Returns:
            v: predictions for all documents for the given concept
        """
        con = self.concepts_with_classifiers[concept_index]
        clf = self.kwd_clfs[cat, con]
        try:  # TODO: explicit option for this rather than interpreting?
            os.fspath(clf)
            clf = joblib.load(clf)["best_estimator_"]
        except TypeError:
            pass
        v = clf.predict_proba(feature_matrix)[:, 1]
        if pbar is not None:
            pbar.update(1)
        return v

    def _predict_kwds_for_cat(
        self,
        feature_matrix: np.array,
        cat_index: int,
        predictions: np.array,
        cat_indices: Dict[str, List[int]] = None,
        use_dask: bool = True,
        pbar: tqdm = None,
    ):
        """
        Make predictions for all documents for all concepts
            in the given category

        Args:
            feature_matrix: array of features for each document
            cat_index: index in categories attribute of the given category
            predictions: the h5 dataset where predictions are stored
            cat_indices: Predicted indices where categories occur
                for each category
            use_dask: Use dask for multiprocessing
            pbar: tqdm progress bar
        """
        cat = self.categories[cat_index]
        pbar.set_postfix(category=cat, refresh=False)
        if (cat_indices is not None) and (cat != ""):
            feature_matrix_test = feature_matrix[cat_indices[cat], :]
            # this could be a problem if I want everything to perfectly align.
        else:
            feature_matrix_test = feature_matrix
        if feature_matrix_test.shape[0] == 0:
            pbar.update(len(self.cat_concept_indices[cat_index]))
            return 0
        # TODO: for good bar, should walk tasks to compute total
        cat_concept_cols = self.cat_concept_indices[cat_index]
        # use the np.where here, bool index for initial setting?
        if False:  # use_dask is True:
            feature_matrix_test = dask.delayed(feature_matrix_test)
            jobs = []
            ProgressBar().register()
            for concept_index in cat_concept_cols:
                j = dask.delayed(self._predict_one_clf)(
                    feature_matrix_test, concept_index, cat, pbar
                )
                jobs.append(j)
            vals = dask.compute(jobs)[0]
        else:
            vals = []
            for concept_index in cat_concept_cols:
                val = self._predict_one_clf(
                    feature_matrix_test, concept_index, cat, pbar
                )
                vals.append(val)
        if (cat_indices is not None) and (cat is not ""):
            # need to correct indices, zeros in places with no predictions
            # TODO: determine if this patching activity
            #  takes longer than just predicting on more
            new_vals = []
            for v in vals:
                new_v = np.zeros(feature_matrix.shape[0])
                new_v[cat_indices[cat]] = v
                new_vals.append(new_v)
            vals = new_vals
        # TODO: below will not work with cat_inds
        if len(vals) > 0:
            topic_preds_sub = np.stack(vals, axis=1)
            predictions[cat_index, :, cat_concept_cols] = topic_preds_sub

    def _predict_keywords(
        self,
        feature_matrix: np.array,
        store: str,
        cat_indices: Dict[str, list] = None,
        only_no_topic: bool = False,
        use_dask: bool = True,
    ):
        """
        Make keyword predictions

        Args:
            feature_matrix: array of features for each document
            store: location of h5 store for predictions
            cat_indices: Predicted indices where categories
                occur for each category
            only_no_topic: only use the models which are
                not associated with a category
            use_dask: use dask for multiprocessing

        Returns:
            store: the location of the h5 store
        """
        all_con_checks = np.sum(
            np.array([a.shape[0] for a in self.cat_concept_indices])
        )
        if Path(store).exists():
            ValueError(f"{store} already exists.")
        with h5py.File(store, "w") as f0, tqdm(total=all_con_checks) as pbar:
            predictions = f0.create_dataset(
                "predictions",
                (
                    len(self.categories),
                    feature_matrix.shape[0],
                    len(self.concepts_with_classifiers),
                ),
                compression="gzip",
            )  # [categories, docs, concepts]
            if only_no_topic is True:
                cat_index = self.categories.index("")
                self._predict_kwds_for_cat(
                    feature_matrix, cat_index, predictions, cat_indices, use_dask, pbar,
                )
            else:
                for cat_index in range(len(self.categories)):
                    self._predict_kwds_for_cat(
                        feature_matrix,
                        cat_index,
                        predictions,
                        cat_indices,
                        use_dask,
                        pbar,
                    )
            return store

    def get_synth_preds(
        self,
        store: str,
        all_cat_inds: Dict[str, np.array],
        batch_size: int,
        only_cat: bool,
        synth_strat: str,
        use_dask: bool = True,
    ) -> np.array:
        """
        Synthesize all keyword models into a single prediction score.

        Args:
            store: location of h5 database
            all_cat_inds: dictionary with keys which are categories.
                Values are index of documents which apply to each category.
            batch_size: batch size for synthesizing predictions
            only_cat: only use category classifiers in synthesis
            synth_strat: strategy for synthesizing category predictions
            use_dask: use dask for multiprocessing

        """
        # TODO: do this without all of the intermediaries
        with h5py.File(store, "r") as f0:
            tdocs = f0["predictions"].shape[1]
            shape = f0["predictions"].shape[1:]
        with tqdm(total=tdocs) as pbar:
            get_synth_preds(
                store,
                shape,
                all_cat_inds,
                self.categories,
                batch_size,
                only_cat,
                synth_strat,
                use_dask,
                pbar=pbar,
            )
        with h5py.File(store, "r") as f0:
            results = f0["synthesis"].value  # TODO: optional return?
        return results

    @staticmethod
    def _to_strings(tags, preds, t):
        all_tag_vals = [
            get_tag_vals(preds[i], tags, t) for i in tqdm(range(preds.shape[0]))
        ]
        return all_tag_vals

    def predict(
        self,
        feature_matrix: np.array,
        cat_threshold: float = 0.5,
        concept_threshold: float = 0.5,
        no_categories: bool = False,
        only_cat: bool = False,
        synth_strat: str = "mean",
        batch_size: int = 10_000,
    ) -> Tuple[PRED_LIST_TYPE, PRED_LIST_TYPE]:
        """
        Make predictions for all input texts.

        Args:
            texts: input texts for which to produce predictions
            cat_threhold: threshold over which to mix in category subset
                model predictions
            concept_threhold: threshold over which to return
                a concept prediction
            no_categories: whether or not to use category-specific models
            only_cat: only use category classifiers in synthesis
            synth_strat: strategy for synthesizing category concept models
                to produce single result.
            batch_size: size of batches for making predictions

        Returns:
            concept_preds: concepts and their belief scores

        Examples:
            >>> examples = ["Olympus Mons is the largest volcano in the solar system",
            ...             "Database management is critical for information retrieval",
            ...             "We used a logistic regression with batched stochastic gradient descent."]
            >>> weights = {'NOUN': 1, 'PROPN': 1, 'ENT': 1, 'NOUN_CHUNK':1, 'ACRONYM': 1}
            >>> features, feature_matrix = hclf.vectorize(examples, weights)
            >>> hclf.predict(feature_matrix)
        """
        n_splits = ceil(feature_matrix.shape[0] / batch_size)
        r1s = []
        # TODO: make temp folder and then write the file
        with NamedTemporaryFile() as tmp_dir:
            tmp_store = Path(f"{tmp_dir.name}/store.h5")
            cat_pred_strings = []
            for n in tqdm(range(n_splits)):
                # TODO: Leave batching to lower methods?
                start = n * batch_size
                end = (n + 1) * batch_size
                matrix_slice = feature_matrix[start:end, :]
                cat_preds = self.predict_categories(matrix_slice)
                cat_inds = get_cat_inds(self.categories, cat_preds, t=cat_threshold)
                LOG.info(f"Predicting keywords")
                store_loc = self._predict_keywords(
                    matrix_slice,
                    tmp_store.name,
                    cat_indices=cat_inds,
                    use_dask=False,
                    only_no_topic=no_categories,
                )
                if no_categories is True:
                    with h5py.File(store_loc) as f0:
                        sp = f0["predictions"][-1, :, :]
                else:
                    LOG.info(f"Synthesizing for each doc.")
                    sp = self.get_synth_preds(
                        store_loc,
                        cat_inds,
                        1000000000,  # TODO: more explanation here
                        only_cat,
                        synth_strat,
                        use_dask=False,
                    )
                LOG.info(f"Converting to strings.")
                r1 = self._to_strings(
                    self.concepts_with_classifiers, sp, concept_threshold
                )
                cp = self._to_strings(self.categories, cat_preds, t=0.0)
                r1s.append(r1)
                cat_pred_strings.append(cp)
            concept_preds = [doc_preds for r1 in r1s for doc_preds in r1]
            all_cat_pred_strings = [
                doc_preds for cp in cat_pred_strings for doc_preds in cp
            ]
        return all_cat_pred_strings, concept_preds


def get_tag_vals(pred_vals: List[float], tags: List[str], t: float):
    tag_vals = [(tags[i], v) for i, v in enumerate(pred_vals) if v > t]
    tag_vals.sort(key=lambda x: -x[1])
    return tag_vals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Use category and concept models to get metrics on the test data."
    )
    parser.add_argument("--experiment_name", help="experiment to generate metrics for")
    parser.add_argument("--out_store", help="h5 store in which to store results")
    parser.add_argument(
        "--out_cat_preds", help="output npy file for category predictions"
    )
    parser.add_argument(
        "--batch_size", help="size of batches for creating ground truth data", type=int,
    )
    parser.add_argument(
        "--limit",
        help="size limit for test data (for testing on smaller subset)",
        type=int,
        default=None,
    )
    args = parser.parse_args()
    main(
        args.experiment_name,
        args.out_store,
        args.out_cat_preds,
        args.batch_size,
        args.limit,
    )
