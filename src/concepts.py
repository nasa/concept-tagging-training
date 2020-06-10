import argparse
import logging

import dsconcept.model as ml

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def main(
    in_corpus,
    concept_field,
    cat_field,
    out_indices,
    out_cat_indices,
    out_raw2lemma,
    out_cat_raw2lemma,
):
    LOG.info(f"Corpus: {in_corpus}")
    LOG.info(f"Keyword Field: {concept_field}")
    LOG.info(f"Category Field: {cat_field}")

    ce = ml.ConceptExtractor()
    ce.from_corpus(in_corpus, concept_field)
    LOG.info(f"Output keyword indices: {out_indices}")
    LOG.info(f"Output keyword raw2lemma: {out_raw2lemma}")
    ce.to_jsons(out_indices, out_raw2lemma)

    LOG.info(f"Extracting categories.")
    ce_higher = ml.ConceptExtractor()
    ce_higher.from_corpus(in_corpus, cat_field)
    LOG.info(f"Output category indices: {out_cat_indices}")
    LOG.info(f"Output category raw2lemma: {out_cat_raw2lemma}")
    ce_higher.to_jsons(out_cat_indices, out_cat_raw2lemma)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Get indices of processed corpus for all of concept and category 
        tags. Also get lemmas for these concepts and categories. Output all of this
        information to json files."""
    )
    parser.add_argument("i", help="input processed jsonlines corpus")
    parser.add_argument("k", help="concept field")
    parser.add_argument("c", help="concept field")
    parser.add_argument("ok", help="output indices for concepts")
    parser.add_argument("oc", help="output indices for categories")
    parser.add_argument("rk", help="out keyword raw to lemma mapping")
    parser.add_argument("rc", help="out category raw to lemma mapping")
    args = parser.parse_args()
    main(args.i, args.k, args.c, args.ok, args.oc, args.rk, args.rc)
