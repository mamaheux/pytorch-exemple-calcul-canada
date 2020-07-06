class AccuracyMetric:
    def __init__(self):
        self._good = 0
        self._total = 0

    def clear(self):
        self._good = 0
        self._total = 0

    def add(self, predicted_class_scores, target_classes):
        predicted_classes = predicted_class_scores.argmax(dim=1)

        self._good += (predicted_classes == target_classes).sum().item()
        self._total += target_classes.size()[0]

    def get_accuracy(self):
        return self._good / self._total
