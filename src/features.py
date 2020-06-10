import argparse
import logging

import dsconcept.model as ml
from multiprocessing import cpu_count

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

N_CPUS = cpu_count()
BATCH_SIZE = 1000


def main(in_corpus, abstract_field, out_features, batch_size, n_threads):
    LOG.info(f"Extracting features from corpus at {in_corpus}.")
    LOG.info(f"Using field: {abstract_field}.")
    fe = ml.FeatureExtractor()
    LOG.info(f"Using batch_size {batch_size} with {n_threads} threads.")
    LOG.info(f"Outputting processed features to {out_features}.")
    fe.from_corpus_to_jsonlines(
        in_corpus, out_features, abstract_field, batch_size, n_threads
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Create features for each document in the processed corpus.
        Each line in output file is a json formatted string 
        with features and their types."""
    )
    parser.add_argument("i", help="input jsonlines corpus")
    parser.add_argument("f", help="abstract field")
    parser.add_argument("o", help="ouput jsonlines features")
    parser.add_argument(
        "-b", help="batch size for feature processing", default=BATCH_SIZE
    )
    parser.add_argument(
        "-n", help="number of threads for features processing", default=N_CPUS
    )
    args = parser.parse_args()
    main(args.i, args.f, args.o, args.b, args.n)
