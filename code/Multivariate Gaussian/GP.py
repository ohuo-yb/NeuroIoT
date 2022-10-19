"""coding:utf-8"""

import math
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Create Data
def create_data():
    val_true = np.loadtxt("dataset/new_power/biLSTM_validation2_True.txt")  # true data
    val_pre = np.loadtxt("dataset/new_power/biLSTM_validation2_Prediction.txt")  # prediction
    val_y = np.loadtxt("dataset/new_power/validation2.txt")[::8][:, 1][84:-1]  # true label

    errors = np.abs(val_true - val_pre)

    df = pd.DataFrame(columns=["errors", "label"])
    df["errors"] = errors
    df["errors"] = df["errors"].ewm(span=2, adjust=True).mean()
    df['label'] = val_y
    data = np.array(df.iloc[:, :])
    print(df)
    return data[:, :-1], data[:, -1]


# Estimation of probabilities using Gaussian distribution
class NaiveBayes:
    def __init__(self):
        self.model = None

    # Mathematical expectations
    @staticmethod
    def mean(X):
        return sum(X) / float(len(X))

    # Standard deviation (variance)
    def stdev(self, X):
        avg = self.mean(X)
        return math.sqrt(sum([pow(x - avg, 2) for x in X]) / float(len(X)))

    # Probability density function    Replace probability with probability density
    def gaussian_probability(self, x, mean, stdev):
        if (2 * math.pow(stdev, 2)) != 0:
            exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
        else:
            exponent = 0
        if exponent == 0:
            return 0
        else:
            return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    def summarize(self, train):
        """
        :param train [[1, 2, 3], [4, 5, 6]]
        """

        s = [(self.mean(i), self.stdev(i)) for i in zip(*train)]
        return s

    def fit(self, X, y):
        """
        :param X Arrays [[1, 2, 3], [1, 2, 3]]
        :param y label [0, 1]
        """
        labels = list(set(y))
        data = {label: [] for label in labels}

        for x, label in zip(X, y):
            data[label].append(x)

        self.model = {label: self.summarize(value)
                      for label, value in data.items()}

        for label, value in data.items():
            self.model[label].append(len(value) / len(X))

        print('train done!')

    def cal_probabilities(self, input_data):
        """
        Calculate the likelihood of each category
        """
        probabilities = {}
        for label, value in self.model.items():
            probabilities[label] = 1
            #  p(y)
            py = value[-1]
            for i in range(len(input_data)):
                _mean, _stdev = value[i]
                #  p(y|x)
                probabilities[label] *= self.gaussian_probability(
                    input_data[i], _mean, _stdev)
            probabilities[label] *= py
        return probabilities

    def predict(self, input_data):
        label = sorted(self.cal_probabilities(
            input_data).items(), key=lambda x: x[-1])[-1][0]
        return label

    def score(self, X, y):
        r = 0.0
        for x, real_y in zip(X, y):
            predict_y = self.predict(x)
            print(predict_y, real_y)
            if predict_y == real_y:
                r += 1

        return r / len(y)


def main():
    X, y = create_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = NaiveBayes()
    model.fit(X_train, y_train)

    print(model.score(X_test, y_test))


if __name__ == "__main__":
    main()
