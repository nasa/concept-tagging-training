import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def main(in_records, inds_loc, out_records_dir):
    train_inds_loc = inds_loc / "train_inds.npy"
    test_inds_loc = inds_loc / "test_inds.npy"
    train_bert_inds_loc = inds_loc / "train_bert_inds.npy"
    dev_bert_inds_loc = inds_loc / "dev_bert_inds.npy"

    LOG.info(f"Loading cleaned records from  {in_records}.")
    records = pd.read_json(in_records, orient="records", lines=True)
    train_inds = np.load(train_inds_loc)
    test_inds = np.load(test_inds_loc)

    LOG.info(f"Creating bert cat df format.")
    mlb = MultiLabelBinarizer()
    cat_bin_array = mlb.fit_transform(records["categories"])
    cat_df = pd.DataFrame(cat_bin_array)
    cat_df.columns = mlb.classes_
    cat_df["text"] = records["text"]
    ordered_cols = ["text"] + mlb.classes_.tolist()
    cat_df = cat_df[ordered_cols]

    train_bert_inds, dev_bert_inds = train_test_split(train_inds, test_size=0.25)
    np.save(train_bert_inds_loc, train_bert_inds)
    np.save(dev_bert_inds_loc, dev_bert_inds)

    ml_sets = {
        "train": cat_df.iloc[train_bert_inds],
        "test": cat_df.iloc[test_inds],
        "dev": cat_df.iloc[dev_bert_inds],
    }

    out_records_dir.mkdir(exist_ok=True)
    for set_type, ml_set in ml_sets.items():
        outfile = out_records_dir / f"{set_type}.csv"
        LOG.info("Writing to {}".format(outfile))
        ml_set.to_csv(outfile, index=True)

    out_id_to_label = str(out_records_dir / "id_to_label.json")
    out_label_to_id = str(out_records_dir / "label_to_id.json")
    out_classes = str(out_records_dir / "classes.txt")

    id_to_label = {i: c for i, c in enumerate(mlb.classes_)}
    label_to_id = {c: i for i, c in enumerate(mlb.classes_)}

    LOG.info(f"Writing classes to {out_classes}")
    classes = mlb.classes_.tolist()

    with open(out_classes, "w") as f0:
        for i, c in enumerate(classes):
            f0.write(c.strip())
            if i < len(classes) - 1:
                f0.write("\n")

    LOG.info(f"Writing to {out_id_to_label}.")
    with open(out_id_to_label, "w") as f0:
        json.dump(id_to_label, f0)

    LOG.info(f"Writing to {out_label_to_id}.")
    with open(out_label_to_id, "w") as f0:
        json.dump(label_to_id, f0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Say hello")
    parser.add_argument("i", help="input records", type=Path)
    parser.add_argument(
        "inds_loc", help="directory for train, test, and dev indices", type=Path
    )
    parser.add_argument(
        "o", help="output files for bert category classifying.", type=Path
    )
    args = parser.parse_args()
    main(args.i, args.inds_loc,  args.o)
