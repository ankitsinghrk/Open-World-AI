
class LabelProvider:

    def __init__(self, labels):
        if not isinstance(labels, dict):
            raise ValueError("Labels must be a dictionary")

        self.labels = labels

    def get_label(self, sample_id):
        if sample_id not in self.labels:
            raise KeyError("Sample ID not found")

        return self.labels[sample_id]
