"""
phishing_detector.py

PhishingDetector class - wraps CountVectorizer + a supervised classifier
to detect phishing vs. legitimate emails.

Follows patterns from the course notes:
- composition (this class "has-a" vectorizer and "has-a" model)
- encapsulation with private attributes + getters
- custom exception when used out of order (ModelNotTrainedError)
- __str__ dunder method
"""

import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from exceptions import ModelNotTrainedError
from email_dataset import EmailDataset


class PhishingDetector:
    """ Machine-learning phishing email detector.

        Pipeline: raw text -> CountVectorizer -> numeric feature matrix (X)
                  -> supervised classifier -> phishing (1) / legit (0)
    """

    def __init__(self, model_name='PhishNet-AI'):
        self.__model_name = model_name
        self.__vectorizer = CountVectorizer()
        self.__classifier = MultinomialNB()
        self.__is_trained = False
        self.__accuracy = None

    # ------------------------------ training -------------------------------
    def train(self, dataset, test_size=0.2, random_state=42):
        """ Trains the model on an EmailDataset instance and evaluates it
            on a held-out test split. Returns the test accuracy. """
        if not isinstance(dataset, EmailDataset):
            raise TypeError('train() expects an EmailDataset instance')

        texts = dataset.getTexts()
        labels = dataset.getLabels()

        X_train_text, X_test_text, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state,
            stratify=labels)

        # fit_transform learns the vocabulary AND builds the feature matrix
        X_train = self.__vectorizer.fit_transform(X_train_text)
        # transform (no fit) re-uses the SAME vocabulary for the test set
        X_test = self.__vectorizer.transform(X_test_text)

        self.__classifier.fit(X_train, y_train)
        self.__is_trained = True

        predictions = self.__classifier.predict(X_test)
        self.__accuracy = accuracy_score(y_test, predictions)

        self.__last_report = classification_report(y_test, predictions,
                                                      target_names=['legit', 'phishing'])
        self.__last_confusion = confusion_matrix(y_test, predictions)

        return self.__accuracy

    # ------------------------------ prediction ------------------------------
    def predict(self, email_text):
        """ Predicts a single email. Returns ('phishing'|'legitimate', confidence). """
        self.__require_trained()
        X = self.__vectorizer.transform([email_text])
        prediction = self.__classifier.predict(X)[0]
        probability = self.__classifier.predict_proba(X)[0]
        label = 'phishing' if prediction == 1 else 'legitimate'
        confidence = probability[prediction]
        return label, confidence

    def predict_batch(self, email_texts):
        """ Predicts a list of emails, returns a list of (label, confidence) tuples. """
        self.__require_trained()
        return [self.predict(text) for text in email_texts]

    # ------------------------------ persistence -----------------------------
    def save_model(self, filepath='phishnet_model.joblib'):
        """ Packages vectorizer + classifier together with Joblib. """
        self.__require_trained()
        bundle = {
            'vectorizer': self.__vectorizer,
            'classifier': self.__classifier,
            'model_name': self.__model_name,
            'accuracy': self.__accuracy
        }
        joblib.dump(bundle, filepath)
        return filepath

    def load_model(self, filepath='phishnet_model.joblib'):
        """ Loads a previously saved vectorizer + classifier bundle. """
        bundle = joblib.load(filepath)
        self.__vectorizer = bundle['vectorizer']
        self.__classifier = bundle['classifier']
        self.__model_name = bundle['model_name']
        self.__accuracy = bundle['accuracy']
        self.__is_trained = True

    # ------------------------------ private helper --------------------------
    def __require_trained(self):
        if not self.__is_trained:
            raise ModelNotTrainedError()

    # ------------------------------ getters ----------------------------------
    def getAccuracy(self):
        return self.__accuracy

    def getModelName(self):
        return self.__model_name

    def isTrained(self):
        return self.__is_trained

    def getConfusionMatrix(self):
        self.__require_trained()
        return self.__last_confusion

    def getClassificationReport(self):
        self.__require_trained()
        return self.__last_report

    # ------------------------------ dunder -----------------------------------
    def __str__(self):
        status = '{:.2%} accuracy'.format(self.__accuracy) if self.__is_trained else 'not trained'
        return 'PhishingDetector[{}] - {}'.format(self.__model_name, status)


if __name__ == '__main__':
    dataset = EmailDataset('emails.csv')
    print(dataset)

    detector = PhishingDetector()
    acc = detector.train(dataset)
    print(detector)
    print('\nClassification report:\n', detector.getClassificationReport())

    detector.save_model('phishnet_model.joblib')

    sample = "Congratulations! You've won a $1000 gift card. Click here now to claim your prize before it expires!"
    label, confidence = detector.predict(sample)
    print('\nSample email -> {} ({:.2%} confidence)'.format(label, confidence))
