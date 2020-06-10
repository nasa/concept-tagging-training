import argparse
import logging
import json

import pandas as pd

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def main(infile, in_subj_mapping, outfile):
    LOG.info(f"Reading corpus from {infile}.")
    df = pd.read_json(infile, orient="records", lines=True)
    LOG.info(f"Shape of input: {df.shape}")

    with open(in_subj_mapping, "r") as f0:
        subj_mapping = json.load(f0)

    def get_subjs(x):
        if type(x) == list:
            s = set(
                subj_mapping[s.strip().lower()]
                for s in x
                if s.strip().lower() in subj_mapping
            )
            l = list(s)
        else:
            l = None
        return l

    categories = df["D072B (Subject Category)"].apply(get_subjs)

    text_col = (
        "<TITLE> " + df["D245A (Title)"] + " <ABSTRACT> " + df["D520B (Abstract)"]
    )
    keywords = (
        df["D650A (NASA Major Indexing Terms)"]
        + df["D659A (NASA Minor Indexing Terms)"]
    )

    pdf = pd.DataFrame()
    pdf["text"] = text_col
    pdf["keywords"] = keywords
    pdf["subjects"] = df["D072B (Subject Category)"]
    pdf["categories"] = categories

    def remove_no_abstracts(x):
        if type(x) == str:
            if "no abstract available" not in x.lower():
                val = True
            else:
                val = False
        else:
            val = False
        return val

    has_abs = pdf["text"].apply(remove_no_abstracts)
    has_kwds = pdf["keywords"].apply(lambda x: type(x) is list)
    has_subj = pdf["keywords"].apply(lambda x: type(x) is list)  # Should be subjects?
    has_cats = pdf["categories"].apply(lambda x: type(x) is list)
    tf = has_kwds & has_subj & has_cats & has_abs
    LOG.info(f"Removed {sum(~tf)} rows.")

    LOG.info(f"Outputting processed corpus to {outfile}.")
    pdf[tf].to_json(outfile, orient="records", lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Merge text and keyword fields from input corpus. 
        Remove documents without abstracts, keywords, or categories."""
    )
    parser.add_argument("i", help="input corpus")
    parser.add_argument("m", help="subject to category mapping json")
    parser.add_argument("o", help="output processed data")
    args = parser.parse_args()
    main(args.i, args.m, args.o)
