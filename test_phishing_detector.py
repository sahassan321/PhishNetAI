"""
test_phishing_detector.py

Unit tests for PhishNet AI, following the same style as test_stats.py
and test_name_format.py from the course notes: setUp() creates shared
fixtures, individual test_ methods check specific behavior.


"""

import unittest
from email_dataset import EmailDataset
from phishing_detector import PhishingDetector
from exceptions import InvalidDatasetError, ModelNotTrainedError


class TestEmailDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = EmailDataset('emails.csv')

    def test_dataset_loads(self):
        self.assertGreater(len(self.dataset), 0)

    def test_missing_file_raises(self):
        with self.assertRaises(InvalidDatasetError):
            EmailDataset('does_not_exist.csv')

    def test_spam_and_legit_counts_add_up(self):
        total = self.dataset.getSpamCount() + self.dataset.getLegitCount()
        self.assertEqual(total, len(self.dataset))

    def test_getitem_returns_text_and_label(self):
        text, label = self.dataset[0]
        self.assertIsInstance(text, str)
        self.assertIn(label, (0, 1))


class TestPhishingDetector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # train once for the whole test class (expensive operation)
        cls.dataset = EmailDataset('emails.csv')
        cls.detector = PhishingDetector()
        cls.detector.train(cls.dataset)

    def test_model_is_trained(self):
        self.assertTrue(self.detector.isTrained())

    def test_accuracy_meets_threshold(self):
        # project claims ~98% accuracy; require at least 90% as a regression guard
        self.assertGreaterEqual(self.detector.getAccuracy(), 0.90)

    def test_predict_returns_valid_label(self):
        label, confidence = self.detector.predict('Hello, are we still meeting for lunch tomorrow?')
        self.assertIn(label, ('phishing', 'legitimate'))
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_obvious_phishing_email_flagged(self):
        phishing_text = ('URGENT: your account has been suspended. Click here '
                          'immediately and verify your password to avoid closure.')
        label, _ = self.detector.predict(phishing_text)
        self.assertEqual(label, 'phishing')

    def test_predict_before_training_raises(self):
        untrained_detector = PhishingDetector()
        with self.assertRaises(ModelNotTrainedError):
            untrained_detector.predict('test email')

    def test_predict_batch_returns_same_length(self):
        emails = ['Meeting at 3pm', 'You won a free iPhone, click now!']
        results = self.detector.predict_batch(emails)
        self.assertEqual(len(results), len(emails))


if __name__ == '__main__':
    unittest.main()
