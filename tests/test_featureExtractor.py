from unittest import TestCase
from dsconcept.model import FeatureExtractor

# from .context import dsconcept
from testfixtures import TempDirectory
from hypothesis import given
import hypothesis.strategies as st
import pytest


@st.composite
def features(draw):
    tags = [st.just(tag) for tag in ["NOUN", "PROPN", "NOUN_CHUNK", "ENT"]]
    tags_strat = st.one_of(*tags)
    txt = st.text(max_size=5)
    doc_feats = st.dictionaries(keys=txt, values=tags_strat, min_size=4, max_size=5)
    feats = draw(st.lists(doc_feats, max_size=5))
    return feats


@st.composite
def weights(draw):
    tags = [st.just(tag) for tag in ["NOUN", "PROPN", "NOUN_CHUNK", "ENT"]]
    tags_strat = st.one_of(*tags)
    weights_dict = draw(
        st.dictionaries(keys=tags_strat, values=st.integers(min_value=0))
    )
    return weights_dict


class TestFeatureExtractor(TestCase):
    def setUp(self):
        self.fe = FeatureExtractor()
        self.d = TempDirectory()
        data = b'{"abstract":"Astronauts are very cool.", "concept": ["ASTRONAUTS", "COOL THINGS"]} \n {"abstract":"NASA is going to Mars.", "concept":["NASA", "MARS"]}'
        self.d.write("test.json", data)
        self.corpus_path = f"{self.d.path}/test.json"

    @given(features())
    def test_features(self, d):
        self.fe.features = d
        self.assertEqual(len(self.fe.features), len(d))

    def test_from_corpus_to_jsonlines(self):
        self.fe.from_corpus_to_jsonlines(
            self.corpus_path, f"{self.d.path}/features.jsonl", "abstract",
        )

    def test_from_jsonlines(self):
        data = b'{"astronaut":"NOUN", "space": "NOUN", "NASA": "ENT"}\n{"Mars": "PROPN", "dog": "NOUN"}'
        features_out = "features.jsonl"
        self.d.write(features_out, data)
        self.fe.from_jsonlines(f"{self.d.path}/{features_out}")
        self.assertSetEqual(self.fe.term_types, {"NOUN", "PROPN", "ENT"})

    def test_to_jsonlines(self):
        self.fe.features = [
            {"space": "NOUN", "Mars": "PROPN"},
            {"Anita": "PROPN", "Adams": "PROPN"},
        ]
        out_features = "features.jsonl"
        self.fe.to_jsonlines(f"{self.d.path}/{out_features}")

    @given(features(), weights())
    def test_weight_terms(self, d, w):
        self.fe.features = d
        self.fe.weight_terms(w)

    @given(features(), weights())
    def test_limit_features(self, d, w):
        self.fe.features = d
        weighted_features = self.fe.weight_terms(
            w
        )  # Test method contingent upon another test. Bad?
        self.fe.limit_features(weighted_features, feature_min=1, feature_max=0.90)

    def tearDown(self):
        self.d.cleanup()
