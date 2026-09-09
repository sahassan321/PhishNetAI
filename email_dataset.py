"""
email_dataset.py

EmailDataset class - loads, validates, and provides access to the
phishing/spam email dataset (Kaggle: emails.csv -> columns: text, spam).

Follows patterns from the course notes:
- private attributes + getters/setters   (Date_class_getters_setters.py)
- data validation                        (1.8 dictionary_data_validation.py)
- custom exceptions                      (user_defined_exception.py)
- __len__ / __str__ dunder methods       (calendar_dunder_methods.py)
"""

import pandas as pd
from exceptions import InvalidDatasetError


class EmailDataset:
    """ Wraps a labeled email dataset (text, spam) loaded from CSV. """

    REQUIRED_COLUMNS = ('text', 'spam')

    def __init__(self, filepath):
        self.__filepath = filepath
        self.__dataframe = None
        self.__load_and_validate()

    # ---------------------- private helper method ----------------------
    def __load_and_validate(self):
        """ Private method: loads the CSV and validates its structure.
            (same private-method pattern as Account_with_private_method.py) """
        try:
            df = pd.read_csv(self.__filepath)
        except FileNotFoundError:
            raise InvalidDatasetError('Dataset file not found', self.__filepath)
        except pd.errors.EmptyDataError:
            raise InvalidDatasetError('Dataset file is empty', self.__filepath)

        for col in EmailDataset.REQUIRED_COLUMNS:
            if col not in df.columns:
                raise InvalidDatasetError(
                    "Missing required column '{}'".format(col), self.__filepath)

        if len(df) == 0:
            raise InvalidDatasetError('Dataset contains no rows', self.__filepath)

        # basic cleaning: drop rows with missing text/labels
        df = df.dropna(subset=list(EmailDataset.REQUIRED_COLUMNS))

        # some rows in the raw Kaggle CSV contain stray control characters
        # that can confuse the CSV parser and leave non-numeric junk in the
        # 'spam' column. Coerce to numeric and drop anything that isn't a
        # clean 0/1 label instead of crashing on the whole dataset.
        df['spam'] = pd.to_numeric(df['spam'], errors='coerce')
        malformed_count = df['spam'].isna().sum()
        df = df.dropna(subset=['spam'])
        df['spam'] = df['spam'].astype(int)

        if len(df) == 0:
            raise InvalidDatasetError('No valid rows remained after cleaning', self.__filepath)

        self.__malformed_rows_dropped = int(malformed_count)
        self.__dataframe = df

    # ------------------------- getters/setters ---------------------------
    def getFilepath(self):
        return self.__filepath

    def getDataframe(self):
        return self.__dataframe

    def getTexts(self):
        """ Returns the email text column as a list. """
        return self.__dataframe['text'].tolist()

    def getLabels(self):
        """ Returns the spam/legit label column as a list (1 = phishing, 0 = legit). """
        return self.__dataframe['spam'].tolist()

    def getSpamCount(self):
        return int(self.__dataframe['spam'].sum())

    def getLegitCount(self):
        return len(self.__dataframe) - self.getSpamCount()

    def getMalformedRowsDropped(self):
        return self.__malformed_rows_dropped

    # ---------------------------- dunder methods --------------------------
    def __len__(self):
        return len(self.__dataframe)

    def __getitem__(self, position):
        row = self.__dataframe.iloc[position]
        return row['text'], row['spam']

    def __str__(self):
        return ('EmailDataset({} emails total | {} spam / {} legit) - source: {}'
                .format(len(self), self.getSpamCount(), self.getLegitCount(),
                        self.__filepath))


if __name__ == '__main__':
    dataset = EmailDataset('emails.csv')
    print(dataset)
    print('First email label:', dataset[0][1])
