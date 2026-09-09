"""
main.py

PhishNet AI - driver application.
Menu-driven interface following the pattern in 1.2(while loop).py.


"""

from email_dataset import EmailDataset
from phishing_detector import PhishingDetector
from exceptions import InvalidDatasetError, ModelNotTrainedError

MODEL_FILE = 'phishnet_model.joblib'
DATA_FILE = 'emails.csv'

menu = ('\n--- PhishNet AI ---\n'
        '1. Train model on emails.csv\n'
        '2. Show model info / accuracy\n'
        '3. Check an email (paste text)\n'
        '4. Save trained model to disk\n'
        '5. Load model from disk\n'
        '6. Exit')

EXIT_OPTION = 6


def main():
    detector = PhishingDetector()
    dataset = None
    option = 0

    while option != EXIT_OPTION:
        print(menu)
        raw = input('Enter option here: ')
        if not raw.isdigit():
            print('Invalid option!')
            continue
        option = int(raw)

        if option == 1:
            try:
                dataset = EmailDataset(DATA_FILE)
                accuracy = detector.train(dataset)
                print('Trained on {} emails. Test accuracy: {:.2%}'.format(
                    len(dataset), accuracy))
            except InvalidDatasetError as e:
                print('Could not load dataset:', e)

        elif option == 2:
            print(detector)
            if detector.isTrained():
                print(detector.getClassificationReport())

        elif option == 3:
            text = input('Paste the email text: ')
            try:
                label, confidence = detector.predict(text)
                print('Result: {}  ({:.2%} confidence)'.format(label.upper(), confidence))
            except ModelNotTrainedError as e:
                print(e)

        elif option == 4:
            try:
                path = detector.save_model(MODEL_FILE)
                print('Model saved to', path)
            except ModelNotTrainedError as e:
                print(e)

        elif option == 5:
            detector.load_model(MODEL_FILE)
            print('Model loaded from', MODEL_FILE)

        elif option == 6:
            print('Exiting PhishNet AI')

        else:
            print('Invalid option!')


if __name__ == '__main__':
    main()
