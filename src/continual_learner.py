from cProfile import label
from pyexpat import features


class ContinualLearner:

    def __init__(self, model, update_function):

        if model is None:
            raise ValueError("Model cannot be empty")

        if not callable(update_function):
            raise ValueError("Update function must be callable")

        self.model = model
        self.update_function = update_function

    def learn(self, features, label):

        if not isinstance(features, (list, tuple)) or len(features) == 0:
            raise ValueError("Features cannot be empty")

        if label is None:
            raise ValueError("Label cannot be empty")

        updated_model = self.update_function(
            self.model,
            features,
            label
        )

        if updated_model is None:
            raise ValueError("Update function returned an empty model")

        self.model = updated_model

        return self.model
