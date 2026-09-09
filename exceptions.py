"""

Custom exception classes for the PhishNet AI project.
Follows the same pattern used in the course notes, user_defined_exception.py, number_guessing_game.py)

"""


class InvalidDatasetError(Exception):
    """Raised when the email dataset is missing, empty, or invalid."""

    def __init__(self, message, filepath=None):
        super().__init__(message)   # reuse Exception's __init__ (like InvalidWithdrawal)
        self.message = message
        self.filepath = filepath

    def __str__(self):
        if self.filepath:
            return '{} (file: {})'.format(self.message, self.filepath)
        return self.message


class ModelNotTrainedError(Exception):
    """Raised when a prediction/save is attempted before the model is trained."""

    def __init__(self, message='Model has not been trained yet. Call train() first.'):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message
