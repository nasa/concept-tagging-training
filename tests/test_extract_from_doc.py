import unittest
from dsconcept.model import *

import logging

logging.basicConfig(level=logging.WARNING)
logging.disable(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.WARNING)


class TestExtractFromDoc(unittest.TestCase):
    def setUp(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.doc = nlp(
            """The NASA Scientific and Technical Information (STI) Program was established to support the 
                   objectives of NASA’s missions and research. The Mission of the STI Program is to support the 
                   advancement of aerospace knowledge and contribute to U.S. competitiveness in aerospace research and 
                   development. This program is essential to help NASA avoid duplication of research by sharing 
                   information and to ensure that the U.S. maintains its preeminence in aerospace-related industries 
                   and education. The NASA STI Program acquires, processes, archives, announces, and disseminates 
                   NASA STI and acquires worldwide STI of critical importance to the
                   National Aeronautics and Space Administation (NASA) and the Nation."""
        )
        self.terms_tagged = extract_from_doc(self.doc)

    def test_is_set(self):
        self.assertEqual(dict, type(self.terms_tagged))

    def test_has_terms(self):
        self.assertGreater(len(self.terms_tagged), 0)

    def test_has_all_feature_types(self):
        self.term_types = {term_type for term, term_type in self.terms_tagged.items()}
        LOG.info(self.term_types)
        LOG.info(self.terms_tagged)
        self.assertEqual(
            {"NOUN", "PROPN", "NOUN_CHUNK", "ENT", "ACRONYM"}, self.term_types
        )


if __name__ == "__main__":
    unittest.main()
