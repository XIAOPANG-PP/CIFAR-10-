import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.epsilon = 1e-12
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred, y_true):
        """
        计算交叉熵损失
        :param y_pred: (n_samples, n_classes)
        :param y_true: (n_samples, n_classes)
        """
        self.y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        self.y_true = y_true
        return -np.sum(self.y_true * np.log(self.y_pred), axis=1).mean()

    def backward(self):
        """
        交叉熵损失 + Softmax 的梯度
        """
        grad = (self.y_pred - self.y_true) / self.y_pred.shape[0]
        return grad
