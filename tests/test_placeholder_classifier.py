import unittest

import numpy as np

from edufer.classification.placeholder_classifier import PlaceholderEmotionClassifier


class PlaceholderClassifierTests(unittest.TestCase):
    def test_classifier_returns_placeholder_bored_signal(self) -> None:
        classifier = PlaceholderEmotionClassifier()
        prediction = classifier.classify(np.zeros((32, 32, 3), dtype=np.uint8))

        self.assertEqual(prediction.label, "bored")
        self.assertTrue(prediction.placeholder)
        self.assertIn("engagement", prediction.educational_signal.lower())


if __name__ == "__main__":
    unittest.main()
