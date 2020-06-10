import argparse
import json
import linecache
import logging
from pathlib import Path

import dsbert.dsbert.multilabel as mll
import numpy as np
import pandas as pd
from tqdm import tqdm

import dsconcept.get_metrics as gm

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def init_domain_bert(base_model_dir, finetuned_model_dir, map_loc="cpu"):
    LOG.info("Loading BERT models")
    processor = mll.MultiLabelTextProcessor(finetuned_model_dir)
    clf = mll.BERTMultilabelClassifier(
        processor, bert_model=base_model_dir, do_lower_case=False,
    )
    clf.initialize_devices()
    clf.load_model(
        f"{finetuned_model_dir}/finetuned_pytorch_model.bin", map_location=map_loc,
    )
    return clf, processor


def load_lines_to_df(data_loc, line_inds):
    tmp_records = []
    for i in tqdm(line_inds):
        r_str = linecache.getline(str(data_loc), i + 1)
        r = json.loads(r_str)
        tmp_records.append(r)
    records_df = pd.DataFrame(tmp_records)
    return records_df


def main(
    data_dir, models_dir, reports_dir, base_model_dir, finetuned_model_dir, sample,
):
    test_inds = np.load(data_dir / "test_inds.npy")
    clean_data_loc = data_dir / "abs_kwds.jsonl"

    in_cat_models = models_dir / "categories/models/"
    in_kwd_models = models_dir / "keywords/models/"
    cat_raw2lemma_loc = models_dir / "cat_raw2lemma.json"

    out_preds_loc = reports_dir / "bert_cat_preds.npy"

    LOG.info("Loading models.")
    cat_clfs = gm.load_category_models(in_cat_models)
    cd = gm.load_concept_models(in_kwd_models)
    clf = gm.HierarchicalClassifier(cat_clfs, cd)
    with open(cat_raw2lemma_loc) as f0:
        cat_raw2lemma = json.load(f0)
    # base_model_dir = str(bert_models_dir / "cased_L-12_H-768_A-12")
    # processor_dir = str(bert_models_dir / "processor_dir")
    # finetuned_model_loc = str(
    #     bert_models_dir / "cased_L-12_H-768_A-12/cache/finetuned_pytorch_model.bin"
    # )
    bert_clf, processor = init_domain_bert(base_model_dir, finetuned_model_dir,)

    LOG.info(f'Loading records from "{clean_data_loc}".')
    if sample is not None:
        lines_to_load = test_inds[0:sample]
    else:
        lines_to_load = test_inds
    records_df = load_lines_to_df(clean_data_loc, lines_to_load)

    LOG.info(f"Processing {len(records_df)} records.")
    df_example = pd.DataFrame()
    df_example["test"] = records_df["text"]
    df_example["label"] = 0
    df_example = df_example.reset_index()
    sample_examples = processor._create_examples(df_example, "test")

    LOG.info("Making BERT category predictions.")
    topic_predictions_df = bert_clf.predict(sample_examples)

    LOG.info("Transforming predictions into matrix which aligns with categories.")
    cols = topic_predictions_df.iloc[:, 2:].columns
    only_preds = topic_predictions_df.iloc[:, 2:]
    tcols = [cat_raw2lemma[c] if c in cat_raw2lemma else c for c in cols]
    only_preds.columns = tcols
    only_preds = only_preds[clf.categories[0:-1]]  # don't include '' cat
    only_pred_vals = only_preds.values

    LOG.info(f'Saving results to "{out_preds_loc}".')
    np.save(out_preds_loc, only_pred_vals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Use BERT cat models to get predictions for all test documents"
    )
    parser.add_argument(
        "--data_dir", help="interim data dir for given experiment", type=Path
    )
    parser.add_argument("--models_dir", help="model_dir for experiment", type=Path)
    parser.add_argument("--reports_dir", help="reports dir for experiment", type=Path)
    parser.add_argument("--base_model_dir", help="base bert model dir", type=str)
    parser.add_argument(
        "--finetuned_model_dir",
        help="dir with classes.txt file and finetuned pytorch model",
        type=str,
    )
    parser.add_argument(
        "--sample", help="how many to sample from test inds", type=int, default=None
    )
    args = parser.parse_args()
    main(
        args.data_dir,
        args.models_dir,
        args.reports_dir,
        args.base_model_dir,
        args.finetuned_model_dir,
        args.sample,
    )
