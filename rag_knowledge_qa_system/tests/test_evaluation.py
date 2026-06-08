import unittest

from evaluation import evaluate_cases, hit_at_k, reciprocal_rank


class Document:
    def __init__(self, source):
        self.metadata = {"source": source}


class EvaluationTest(unittest.TestCase):
    def test_hit_at_k(self):
        self.assertEqual(hit_at_k(["a.txt", "b.txt"], "b.txt"), 1)
        self.assertEqual(hit_at_k(["a.txt"], "b.txt"), 0)

    def test_reciprocal_rank(self):
        self.assertEqual(reciprocal_rank(["a.txt", "b.txt"], "b.txt"), 0.5)
        self.assertEqual(reciprocal_rank(["a.txt"], "b.txt"), 0.0)

    def test_evaluate_cases(self):
        def search(_question):
            return [(Document("rules.txt"), 0.9)]

        metrics = evaluate_cases(
            search,
            [{"question": "when?", "expected_source": "rules.txt"}],
        )
        self.assertEqual(metrics["hit_at_k"], 1.0)
        self.assertEqual(metrics["mrr"], 1.0)


if __name__ == "__main__":
    unittest.main()
