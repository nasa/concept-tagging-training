import argparse
import logging
from math import ceil
from pathlib import Path
from time import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import linregress
from tqdm import tqdm

import dsconcept.get_metrics as gm
from dsconcept.get_metrics import get_keyword_results

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def lim_concepts_and_plot(mean_df, tmp_df, fig_dir):
    LOG.info(f"tmp_df.shape={tmp_df.shape}")
    cat = tmp_df["category"].iloc[0]
    lim_mean_df = mean_df[np.in1d(mean_df["concept"], tmp_df["concept"])]
    lim_tmp_df = tmp_df[np.in1d(tmp_df["concept"], mean_df["concept"])]
    if lim_mean_df.shape[0] != lim_tmp_df.shape[0]:
        ValueError("Different df sizes")
    metrics = ["recall", "precision", "f1", "roc_auc"]
    for m in metrics:
        a = 0.3
        lim_tmp_df[m].hist(alpha=a, label=f"one_layer | cat={cat}")
        lim_mean_df[m].hist(alpha=a, label="mean")
        plt.legend()
        plt.title(m)
        fig_loc = fig_dir / f"{m}.png"
        LOG.info(f"Saving plot to {fig_loc}")
        plt.savefig(fig_loc)
        plt.clf()


def load_classifier(in_cat_models, in_kwd_models, in_vectorizer):
    LOG.info(f"Loading category classifiers from {in_cat_models}.")
    in_clfs = list(in_cat_models.iterdir())
    cat_clfs = [joblib.load(c) for c in tqdm(in_clfs)]

    LOG.info(f"Loading keyword classifiers from {in_kwd_models}.")
    cd = {}  # expects no_topics with suffix ''
    for topic_dir in tqdm(in_kwd_models.iterdir()):
        in_clfs = list(topic_dir.iterdir())
        clfs = [joblib.load(c) for c in in_clfs]  # loads the classifiers
        topic_name = topic_dir.stem.split("_")[1]
        # depends on opinionated path format
        for c in clfs:
            cd[topic_name, c["concept"]] = c["best_estimator_"]

    hclf = gm.HierarchicalClassifier(cat_clfs, cd)
    hclf.load_vectorizer(in_vectorizer)
    return hclf


def get_clf_times(hclf, small_res, weights, sizes):
    hl_strats = ["topics", "only_no_topic"]
    batch_size = 10_000_000  # TODO: remove batching
    hl_dicts = []

    for hls in hl_strats:
        times = []
        out_sizes = []
        for s_size in sizes:
            if s_size > small_res.shape[0]:
                LOG.warning(f"Skipping {s_size} because it is greater than data size.")
                continue
            examples = small_res["text"].sample(s_size)
            n_splits = ceil(examples.shape[0] / batch_size)
            t1 = time()
            for n in tqdm(range(n_splits)):
                start = n * batch_size
                end = (n + 1) * batch_size
                example_batch = examples[start:end]
                _, feature_matrix = hclf.vectorize(example_batch, weights)
                LOG.info(f"Predicting keywords")
                if hls == "only_no_topic":
                    no_categories = True
                elif hls == "topics":
                    no_categories = False
                else:
                    LOG.exception(f"Invalid strategy selection: {hls}")
                _, _ = hclf.predict(feature_matrix, 0.5, 0.5, no_categories)
            t2 = time()
            tt = t2 - t1
            out_sizes.append(s_size)
            times.append(tt)
        hld = {
            "strat": hls,
            "times": times,
            "sizes": out_sizes,
        }
        hl_dicts.append(hld)
    return hl_dicts


def make_time_plots(hl_dicts, out_plot_file):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for hd in hl_dicts:
        lg = linregress(hd["sizes"], hd["times"])
        docs_per_sec = [s / t for s, t in zip(hd["sizes"], hd["times"])]
        a = np.array(hd["sizes"])
        axes[0].plot(hd["sizes"], hd["times"], marker="o", label=hd["strat"])
        axes[0].plot(hd["sizes"], lg.slope * a + lg.intercept, "r", alpha=0.5)
        axes[0].set_xlabel("number of docs")
        axes[0].set_ylabel("time to tag (seconds)")
        axes[0].set_title("Time to tag depending on batch size")
        axes[1].plot(hd["sizes"], docs_per_sec, marker="o", label=hd["strat"])
        axes[1].set_xlabel("number of docs")
        axes[1].set_ylabel("tagging rate (docs/seconds)")
        axes[1].set_title("Tagging rate depending on batch size")
        axes[0].legend()
        axes[1].legend()
    plt.savefig(out_plot_file)
    plt.clf()


def main(
    in_mean,
    in_cats_dir,
    in_kwds_dir,
    in_vectorizer,
    in_clean_data,
    in_config,
    out_plots_dir,
):
    LOG.info("Loading dataframes.")
    mean_df = pd.read_csv(in_mean, index_col=0)
    no_synth_df = get_keyword_results(in_kwds_dir)
    if no_synth_df.shape[0] == 0:
        raise ValueError(
            f"No keyword results. Are the subdirectories of {in_kwds_dir} empty?"
        )
    no_cat_df = no_synth_df[no_synth_df["category"] == ""]
    LOG.info("Making plots.")
    lim_concepts_and_plot(mean_df, no_cat_df, out_plots_dir)

    with open(in_config) as f0:
        config = yaml.safe_load(f0)
    hclf = load_classifier(in_cats_dir, in_kwds_dir, in_vectorizer)
    full_corpus = pd.read_json(in_clean_data, orient="records", lines=True)
    sizes = [1, 10, 50, 100, 250, 500, 1000, 2000, 5000, 10_000]
    sample_size = min(max(sizes), full_corpus.shape[0])

    small_res = pd.read_json(in_clean_data, orient="records", lines=True).sample(
        sample_size
    )

    hl_dicts = get_clf_times(hclf, small_res, config["weights"], sizes)

    out_plots_time = out_plots_dir / "time_v_batch_size.png"
    make_time_plots(hl_dicts, out_plot_file=out_plots_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""From output metrics, create plots for ROC-AUC, F1, precision, 
        and recall"""
    )
    parser.add_argument("--mean", help="results from synthesis with max strategy")
    parser.add_argument("--in_cats_dir", help="category classifiers dir", type=Path)
    parser.add_argument("--in_kwds_dir", help="kwds classifier models dir", type=Path)
    parser.add_argument("--in_vectorizer", help="vectorizer location", type=Path)
    parser.add_argument("--in_clean_data", help="clean code location", type=Path)
    parser.add_argument("--in_config", help="config location", type=Path)
    parser.add_argument("--out_plots_dir", help="output dir for plots pngs", type=Path)
    args = parser.parse_args()
    main(
        args.mean,
        args.in_cats_dir,
        args.in_kwds_dir,
        args.in_vectorizer,
        args.in_clean_data,
        args.in_config,
        args.out_plots_dir,
    )
