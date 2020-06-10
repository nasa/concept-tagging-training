from unittest import TestCase
from .context import dsconcept
from testfixtures import TempDirectory
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class TestConceptExtractor(TestCase):
    def setUp(self):
        self.ce = dsconcept.model.ConceptExtractor()
        self.d = TempDirectory()

    def test_concept_sets(self):
        self.ce.concept_sets = [
            ["MARS", "NASA"],
            ["NASA"],
            ["MARS"],
            ["HIT", "JUPITER"],
        ]

    def test_from_corpus(self):
        data = b'{"abstract":"Astronauts are very cool.", "concept": ["ASTRONAUTS", "COOL THINGS"]} \n {"abstract":"NASA is going to Mars.", "concept":["NASA", "MARS"]}'
        self.d.write("test.json", data)
        self.ce.from_corpus(Path(f"{self.d.path}/test.json"), "concept")

    def test_get_top_concepts(self):
        self.ce.concept_sets = [
            ["MARS", "NASA"],
            ["NASA"],
            ["MARS"],
            ["HIT", "JUPITER"],
        ]
        self.assertDictEqual(
            self.ce.get_top_concepts(2), {"mars": [0, 2], "nasa": [0, 1]}
        )

    def tearDown(self):
        self.d.cleanup()
