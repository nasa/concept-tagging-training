"""
model
-----
Program to make classifiers from input corpus and selected keyword field.

Author: Anthony Buonomo
Contact: anthony.r.buonomo@nasa.gov

Classes to support document classification.
"""

from collections import Counter
import logging
from multiprocessing import cpu_count
import json
from typing import Dict
from tqdm import tqdm

import spacy
from spacy.lemmatizer import Lemmatizer
from spacy.lookups import Lookups
from textacy.extract import acronyms_and_definitions

nlp = spacy.load("en_core_web_sm")

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def spacy_tokenizer(txt):
    """
    Tokenize txt using spacy. Fit for use with sklearn CountVectorizer.

    Args:
        txt (str): text to be tokenized

    Returns:
        terms_tagged_list (list of str): tokens extracted from text

    Examples:
        >>> from dsconcept.model import spacy_tokenizer
        >>> from sklearn.feature_extraction.text import CountVectorizer
        >>> txt = "The ship hung in the sky much the same way bricks don't."
        >>> doc_tokens = spacy_tokenizer(txt)
        >>> doc_tokens
        ['ship :: NOUN',
         'sky :: NOUN',
         'way :: NOUN',
         'brick :: NOUN',
         'the ship :: NOUN_CHUNK']
        >>> v = CountVectorizer(txt, tokenizer=spacy_tokenizer)
        >>> v.fit_transform([txt])
        >>> v.vocabulary_
        {'ship :: NOUN': 1, 'sky :: NOUN': 2, 'way :: NOUN': 3, 'brick :: NOUN': 0}
    """
    doc = nlp(txt)
    terms_tagged = extract_from_doc(doc)
    terms_tagged_list = [f"{term} :: {tag}" for term, tag in terms_tagged.items()]
    return terms_tagged_list


def should_keep(w, desired_parts_of_speech):
    desiredPOS = w.pos_ in desired_parts_of_speech
    notStop = not w.is_stop
    notPerc = w.lemma_ not in ["%"]
    return desiredPOS and notStop and notPerc


def extract_from_doc(doc):
    """
    Extract features from a spacy doc.

    Args:
        doc (spacy.doc): a doc processed by the spacy 'en' model

    Returns:
        terms_tagged (dict): features with their respective tags

    Examples:
        >>> from dsconcept.model import extract_from_doc
        >>> import spacy
        >>> nlp = spacy.load('en_core_web_sm')
        >>> txt = "The ship hung in the sky much the same way bricks don't."
        >>> doc = nlp(txt)
        >>> features = extract_from_doc(doc)
        >>> features
        {'ship': 'NOUN',
         'sky': 'NOUN',
         'way': 'NOUN',
         'brick': 'NOUN',
         'the ship': 'NOUN_CHUNK'}
    """
    # TODO: change this function such that it processes better but maintains the same interface.
    terms_tagged = dict()

    desired_parts_of_speech = ["NOUN", "PROPN"]
    # Get any 1-gram terms which are not % signs, or stop words.
    terms = {w.lemma_: w.pos_ for w in doc if should_keep(w, desired_parts_of_speech)}
    terms_tagged.update(terms)

    # Lemmatize each gram and join with a space.
    noun_chunks = {
        " ".join([w.lemma_ for w in nc if not w.is_stop]): nc.label_
        for nc in doc.noun_chunks
    }
    # filter our noun chunks that are already in terms set and not in excluded_list.
    excluded_list = ["-PRON-", ""]
    noun_chunks_filtered = {
        w.strip(): "NOUN_CHUNK"
        for w, lab in noun_chunks.items()
        if (w not in terms.keys()) and (w not in excluded_list)
    }
    terms_tagged.update(noun_chunks_filtered)

    # TODO: entities take precedence over noun chunks
    # Get entities from text and remove collisions with terms and noun chunks.
    ent_excluded_set = ["ORDINAL", "CARDINAL", "QUANTITY", "DATE", "PERCENT"]
    ents = {e.lemma_: e.label_ for e in doc.ents if e.label_ not in ent_excluded_set}
    ents_filtered = {
        ent: "ENT"
        for ent, lab in ents.items()
        if ent not in terms.keys() and ent not in noun_chunks_filtered.keys()
    }
    terms_tagged.update(ents_filtered)

    # Add acronyms which have definitions.
    # These acronyms could create Noise if they are not good. Maybe better to use their definitions.
    # This schema will only pull out identifical definitions. No lemmatizing, no fuzzy matching.
    # TODO: add lemmatizing and fuzzy matching for acrnoyms. This code exists in acronyms project.
    acronyms_with_defs = acronyms_and_definitions(doc)
    acronyms_filtered = {
        "{} - {}".format(ac, definition): "ACRONYM"
        for ac, definition in acronyms_with_defs.items()
        if definition != ""
    }
    terms_tagged.update(acronyms_filtered)

    return terms_tagged


def extract_features_from_abstracts(
    descriptions, feature_outfile, batch_size=1000, n_threads=cpu_count(), total=None
):
    """
    Generate features from input batch of abstracts.

    Args:
        descriptions (list of str): list of descriptions
        feature_outfile (str): output file for features jsonlines
        batch_size (int): how many docs to process in a batch
        n_threads (int): number of threads to process with
        total (int): total number of description to optionally pass to tqdm for a better loading bar

    Returns:
        no_descriptions (int): hown many descriptions were processed

    Examples:
        >>> from dsconcept.model import extract_features_from_abstracts
        >>> import json
        >>>
        >>> abstract1 = " A common mistake that people make when trying to design something completely foolproof is to underestimate the ingenuity of complete fools."
        >>> abstract2 = "Since we decided a few weeks ago to adopt the leaf as legal tender, we have, of course, all become immensely rich."
        >>> abstracts = [abstract1, abstract2]
        >>>
        >>> feature_outfile = 'data/tmp_features.txt'
        >>>
        >>> extract_features_from_abstracts(abstracts, feature_outfile, batch_size=1, n_threads=1)
        >>>
        >>> with open(feature_outfile, 'r') as f0:
        >>>     content = f0.readlines()
        >>> features = [json.loads(line) for line in content]
        >>> features
        [{'mistake': 'NOUN',
          'people': 'NOUN',
          'ingenuity': 'NOUN',
          'fool': 'NOUN',
          'a common mistake': 'NOUN_CHUNK',
          'complete fool': 'NOUN_CHUNK'},
         {'week': 'NOUN',
          'leaf': 'NOUN',
          'tender': 'NOUN',
          'course': 'NOUN',
          'legal tender': 'NOUN_CHUNK'}]
    """

    LOG.info("Extracting features to {}".format(feature_outfile))
    no_descriptions = 0
    with open(feature_outfile, "w") as f0:
        for doc in tqdm(
            nlp.pipe(descriptions, batch_size=batch_size, n_threads=n_threads,),
            total=total,
        ):
            json.dump(extract_from_doc(doc), f0)  # each line is valid json
            f0.write("\n")
            no_descriptions += 1

    LOG.info("Extracted feature sets to {}".format(feature_outfile))
    return no_descriptions


class FeatureExtractor:
    def __init__(self):
        """
        A term extractor.

        Examples:
            >>> from dsconcept.model import FeatureExtractor
            >>> extractor = FeatureExtractor()
        """
        self._features = list()
        self.term_types = dict()
        self.feature_counts = Counter()

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, value):
        self._features = value
        self.term_types = {
            term_type
            for feature_set in self._features
            for term_type in feature_set.values()
        }
        all_features = [
            feature
            for feature_set in self._features
            for feature, val in feature_set.items()
        ]
        self.feature_counts = Counter(all_features)

    @staticmethod
    def from_corpus_to_jsonlines(
        in_corpus, out_features, abstract_field, batch_size=1000, n_threads=cpu_count()
    ):
        """

        Args:
            in_corpus (pathlib.Path | str): input path to json file containing corpus
            out_features (pathlib.Path | str): output path for features json lines file.
            abstract_field (str): name of abstract field for corpus
            batch_size (int): size of batch to use when multithreading using spacy's nlp.pipe
            n_threads (int): number of threads to use when multithreading using spacy's nlp.pipe

        Returns:
            n_descriptions (int): the number of abstracts in the corpus

        """

        n_lines = file_len(in_corpus)
        with open(in_corpus, "r") as f0:
            record_generator = (json.loads(l) for l in f0.readlines())
        text_generator = (r[abstract_field] for r in record_generator)
        n_descriptions = extract_features_from_abstracts(
            text_generator, out_features, batch_size, n_threads, total=n_lines
        )
        return n_descriptions

    def from_jsonlines(self, in_features):
        """
        Load features from jsonlines.

        Args:
            in_features (pathlib.Path | str): path to input jsonlines features file

        Returns:
            in_features (pathlib.Path | str): path to input jsonlines features file

        """
        with open(in_features, "r") as f0:
            content = (
                f0.readlines()
            )  # each line is json formatted, but whole file is not.
        self.features = [json.loads(line) for line in content]
        return in_features

    def to_jsonlines(self, out_features):
        """
        Output features to jsonlines.

        Args:
            out_features (pathlib.Path | str): output path to features jsonlines file

        Returns:
            out_features (pathlib.Path | str): output path to features jsonlines file

        """
        with open(out_features, "w") as f0:
            for feature_set in self.features:
                json.dump(feature_set, f0)  # each line is valid json
                f0.write("\n")
        return out_features

    def weight_terms(self, weights: Dict[str, int]):
        """
        Weights features according to tag type.

        Args:
            weights (dict of str): mappings from term types to their weights

        Returns:
            weighted_features (list of dict): features with mappings to weights instead of term types

        Examples
        --------
        >>> weights = {'NOUN': 1, 'NOUN_CHUNK': 2}
        >>> weighted_features = tm.weight_terms(weights)
        >>> weighted_features
        [{'mistake': 1,
          'people': 1,
          'ingenuity': 1,
          'fool': 1,
          'a common mistake': 2,
          'complete fool': 2},
         {'week': 1, 'leaf': 1, 'tender': 1, 'course': 1, 'legal tender': 2}]
        """
        assert type(weights) is dict, "Weights must be dict: {}".format(weights)
        if self.term_types > weights.keys():
            LOG.warning(
                "Term types without a specified weight will be omitted from returned feature sets."
            )
        elif self.term_types < weights.keys():
            LOG.warning(
                "More term types specified then those which exist in corpus. Ignoring excess."
            )
        weighted_features = [
            weight_terms_inner(doc_features, weights) for doc_features in self.features
        ]
        return weighted_features

    def limit_features(
        self,
        weighted_features,
        feature_min,
        feature_max,
        topic=None,
        doc_topic_matrix=None,
    ):
        """
        Cull features.

        Args:
            weighted_features (list of dict): features with assigned weights
            feature_min (int): features which have in-corpus frequencies under feature_min are excluded.
            feature_max (float): features which occur in greater than this percentage of documents are excluded.
            topic (int | None): if specified, only return feature sets with maximum probability to be in this topic.
            doc_topic_matrix (numpy.ndarray): topic probability distributions for each document in corpus.

        Returns:
            weighted_limited (list): limited features with assigned weights

        Examples:
            >>> limited_features = tm.limit_features_for_X(weighted_features, feature_min=1, feature_max=0.99)
        """
        assert (feature_max > 0.0) and (
            feature_max <= (1.0)
        ), "feature_max should be float in (0,1]"
        feature_ex = {
            feature: occurrence
            for feature, occurrence in self.feature_counts.items()
            if (occurrence >= feature_min)
            and (occurrence / len(self.features) < feature_max)
        }

        weighted_limited = [
            {
                feature: val
                for feature, val in feature_set.items()
                if feature in feature_ex
            }
            for feature_set in weighted_features
        ]

        if topic is not None:
            assert doc_topic_matrix is not None, LOG.error(
                "Must supply doc_topic_matrix when using topic model segmentation."
            )
            LOG.info(f"Segmenting vectorizer and matrix for topic {topic}.")
            print("here")
            in_topic_index = [
                i for i, distr in enumerate(doc_topic_matrix) if distr.argmax() == topic
            ]
            weighted_limited = [weighted_limited[i] for i in in_topic_index]

        return weighted_limited


def weight_terms_inner(doc_features, weights):
    """

    Args:
        doc_features (dict): features with assigned tags
        weights (dict): tag to weight mappings

    Returns:
        weighted_terms (dict): features with assigned weights

    Examples
        >>> from dsconcept.model import weight_terms_inner
        >>> features = {'ship': 'NOUN', 'sky': 'NOUN', 'way': 'NOUN', 'brick': 'NOUN', 'the ship': 'NOUN_CHUNK'}
        >>> weights = {'NOUN': 1, 'NOUN_CHUNK': 3}
        >>> weighted_terms = weight_terms_inner(features, weights)
        >>> weighted_terms
        {'ship': 1, 'sky': 1, 'way': 1, 'brick': 1, 'the ship': 3}
    """
    weighted_terms = {}
    for pos0, weight in weights.items():
        updated_dict = {w: weight for w, pos in doc_features.items() if pos == pos0}
        weighted_terms.update(updated_dict)

    return weighted_terms


class ConceptExtractor:
    def __init__(self):
        """
        Information about relationship between concepts/keywords and corpus.

        Examples:
            >>> from dsconcept.model import ConceptExtractor
            >>> kwd_sets = [['Zaphod', 'Arthur'], ['Arthur'], ['Zaphod'], ['Heart of Gold']]
            >>> info = ConceptExtractor.concept_sets = kwd_sets
            >>> info.concepts
            {'arthur', 'heart of gold', 'zaphod'}
        """
        self._concept_sets = []
        self.raw2lemma = {}
        self.lemma2raw = {}
        self.lemmatizer = None
        self.concepts_frequencies = Counter()
        self.concepts = set()
        self.concept_index_mapping = {}

    @property
    def concept_sets(self):
        return self._concept_sets

    @concept_sets.setter
    def concept_sets(self, value):
        """
        Sets concepts_sets and the attributes derived from it.

        Args:
            value (list of list of str): A list of lists of strings; each string being a concept,
                each set in the larger list corresponding to a document which has the tags seen in the set.
        """
        self._concept_sets = value
        LOG.debug("Extracting raw keywords as concepts.")
        all_concepts = [
            concept
            for concept_set in tqdm(self._concept_sets)
            for concept in concept_set
            if concept.strip() != ""
        ]
        raw_concepts = set(all_concepts)

        LOG.debug("Lemmatizing {} raw concepts.".format(len(raw_concepts)))
        concepts = [c.lower() for c in raw_concepts]

        self.raw2lemma = {rc: c for rc, c in zip(raw_concepts, concepts)}
        lookups = Lookups()
        lookups.add_table("lemma_lookup", self.raw2lemma)
        self.lemmatizer = Lemmatizer(lookups)
        self.lemma2raw = {v: k for k, v in self.raw2lemma.items()}
        lemma_concepts = [
            self.lemmatizer(concept, "NOUN")[0] for concept in all_concepts
        ]
        self.concepts_frequencies = Counter(lemma_concepts)
        self.concepts = set(lemma_concepts)
        self._fit_concept_indices()

    def _fit_concept_indices(self):
        kwd_sets_lemmas = [
            [self.lemmatizer(kwd, "NOUN")[0] for kwd in kwd_set]
            for kwd_set in self.concept_sets
        ]
        concepts_with_inds = dict()
        for i, kwd_set in enumerate(kwd_sets_lemmas):
            for kwd in kwd_set:
                if kwd not in concepts_with_inds:
                    concepts_with_inds[kwd] = [i]
                else:
                    concepts_with_inds[kwd].append(i)
        self.concept_index_mapping = concepts_with_inds

    def from_corpus(self, in_corpus, concept_field):
        """
        Extract concepts from input json corpus.

        Args:
            in_corpus (pathlike): path to input json-formatted corpus from which to extract concepts
            concept_field (str): the name of the concept field
        """
        with open(in_corpus, "r") as f0:
            record_generator = (json.loads(l) for l in f0.readlines())
        concept_sets = [r[concept_field] for r in record_generator]
        with_concepts = [i for i, cs in enumerate(concept_sets) if cs is not []]
        assert len(with_concepts) > 0, LOG.error(
            f'"{concept_field}" not present in corpus.'
        )
        LOG.debug(f"{len(with_concepts)} docs in corpus with {concept_field}.")
        self.concept_sets = concept_sets

    def to_jsons(self, out_indices, out_raw2lemma):
        """
        Output indices and raw2lemma dicts to json files.

        Args:
            out_indices (pathlib.Path): path to output file containing indices for concepts
            out_raw2lemma (pathlib.Path): path to output file containing mappings from concepts to their lemmas

        Returns:
            out_indices (pathlib.Path): path to output file containing indices for concepts
            out_raw2lemma (pathlib.Path): path to output file containing mappings from concepts to their lemmas

        """
        with open(out_indices, "w") as f0:
            json.dump(self.concept_index_mapping, f0)
        with open(out_raw2lemma, "w") as f0:
            json.dump(self.raw2lemma, f0)
        return out_indices, out_raw2lemma

    def from_jsons(
        self, in_indices, in_raw2lemma
    ):  # a little strange because it does not fill in all attributes
        """
        Load index and raw2lemma dictionaries into empty ConceptExtractor

        Args:
            in_indices ():
            in_raw2lemma ():
        """
        with open(in_indices, "r") as f0:
            self.concept_index_mapping = json.load(f0)
        with open(in_raw2lemma, "r") as f0:
            self.raw2lemma = json.load(f0)
        lookups = Lookups()
        lookups.add_table("lemma_lookup", self.raw2lemma)
        self.lemmatizer = Lemmatizer(lookups)
        self.lemma2raw = {v: k for k, v in self.raw2lemma.items()}
        self.concepts = self.concept_index_mapping.keys()
        tmp_frequencies = {
            concept: len(index) for concept, index in self.concept_index_mapping.items()
        }
        self.concepts_frequencies = Counter(tmp_frequencies)

    def get_top_concepts(self, min_freq=500):
        """

        Args:
            min_freq (int): occurrence threshold for concepts

        Returns:
            top_concepts(dict): a subset of the

        Examples:
            >>> info.get_top_concepts(2)
            >>> info.top_concepts
            ['zaphod', 'arthur']
        """
        LOG.info(f"Getting indices for concepts with frequency >= {min_freq}.")
        top_concepts = {
            concept: index
            for concept, index in self.concept_index_mapping.items()
            if len(index) >= min_freq
        }
        return top_concepts
