from sklearn import metrics


class Metric:
    def __init__(self):
        self.predictions = []
        self.labels = []

    def reset(self):
        self.predictions = []
        self.labels = []

    def update(self, predictions, labels):
        predictions = list(predictions.cpu().detach().numpy())
        labels = list(labels.cpu().detach().numpy())
        self.predictions += predictions
        self.labels += labels

    def compute(self):
        predictions = [1 if p >= 0.5 else 0 for p in self.predictions]
        probability = self.predictions
        print(
            f'auc: {metrics.roc_auc_score(self.labels, probability):.4f}\t'
            f'accuracy: {metrics.accuracy_score(self.labels, predictions):.4f}\t'
            f'precision: {metrics.precision_score(self.labels, predictions):.4f}\t'
            f'recall: {metrics.recall_score(self.labels, predictions):.4f}\t'
            f'f1 score: {metrics.f1_score(self.labels, predictions):.4f}'
        )
        return metrics.f1_score(self.labels, predictions)
