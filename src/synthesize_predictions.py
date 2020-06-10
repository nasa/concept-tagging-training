import argparse
import logging
from pathlib import Path

import dask
import h5py
import joblib
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from tqdm import tqdm

from dsconcept.get_metrics import (
    get_cat_inds,
    get_synth_preds,
    load_category_models,
    load_concept_models,
    HierarchicalClassifier,
    get_mets,
)

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def main(
    experiment_name,
    synth_strat,
    in_cat_preds,
    out_store,
    synth_batch_size,
    t,
    out_synth_scores,
    limit=None,
    con_limit=None,
):
    test_inds = np.load(f"data/interim/{experiment_name}/test_inds.npy")
    feature_matrix = joblib.load(f"data/interim/{experiment_name}/feature_matrix.jbl")
    in_cat_models = Path(f"models/{experiment_name}/categories/models/")
    in_kwd_models = Path(f"models/{experiment_name}/keywords/models/")
    cat_preds = np.load(in_cat_preds)  # based on experiment or explicit path?
    cat_clfs = load_category_models(in_cat_models)
    cd = load_concept_models(in_kwd_models)
    clf = HierarchicalClassifier(cat_clfs, cd)

    if limit is not None:
        LOG.info(f"Limiting to {limit} test records.")
        feature_matrix_test = feature_matrix.tocsc()[test_inds[0:limit], :]
        cat_preds = cat_preds[0:limit, :]
        # TODO: How does this affect indices?
    else:
        feature_matrix_test = feature_matrix.tocsc()[test_inds, :]

    LOG.info(f'Synthesizing predictions with strategy "{synth_strat}".')
    all_cat_inds = get_cat_inds(clf.categories, cat_preds, t=t)
    if con_limit is not None:
        conwc = clf.concepts_with_classifiers[0:con_limit]
    else:
        conwc = clf.concepts_with_classifiers
    shape = (feature_matrix_test.shape[0], len(conwc))
    with tqdm(total=shape[0]) as pbar:
        get_synth_preds(
            out_store,
            shape,
            all_cat_inds,
            clf.categories,
            synth_batch_size,
            only_cat=False,
            synth_strat=synth_strat,
            con_limit=con_limit,
            limit=limit,
            pbar=pbar,
        )

    LOG.info("Obtaining metrics.")
    with h5py.File(out_store, "r") as f0:
        if limit is not None:
            target_values = f0["ground_truth"][0:limit, :]
        else:
            target_values = f0["ground_truth"].value
    with h5py.File(out_store, "r") as f0:
        synth_preds = f0["synthesis"].value

    jobs = []
    mets_pbar = tqdm(
        range(len(conwc)),
        total=len(conwc),
    )
    for i in mets_pbar:
        job = dask.delayed(get_mets)(
            i, synth_preds, target_values, conwc, mets_pbar
        )
        jobs.append(job)
    records = dask.compute(jobs)
    new_recs_df = pd.DataFrame(records[0])
    LOG.info(f"Saving results to {out_synth_scores}.")
    new_recs_df.to_csv(out_synth_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Say hello")
    parser.add_argument("--experiment_name", help="input txt file")
    parser.add_argument("--synth_strat", help="input txt file")
    parser.add_argument("--in_cat_preds", help="input txt file")
    parser.add_argument("--store", help="input txt file")
    parser.add_argument("--synth_batch_size", help="input txt file", type=int)
    parser.add_argument("--threshold", help="input txt file", type=float)
    parser.add_argument("--out_synth_scores", help="input txt file")
    parser.add_argument(
        "--limit", help="size for sample to test synthesis", type=int, default=None
    )
    parser.add_argument(
        "--con_limit", help="size for concept sample", type=int, default=None
    )
    args = parser.parse_args()
    main(
        args.experiment_name,
        args.synth_strat,
        args.in_cat_preds,
        args.store,
        args.synth_batch_size,
        args.threshold,
        args.out_synth_scores,
        args.limit,
        args.con_limit,
    )
