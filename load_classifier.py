from abc import ABC, abstractmethod

import joblib


class Classifier(ABC):
    @abstractmethod
    def model_class(self):
        pass


class NBClassifier(Classifier):

    def model_class(self):
        return joblib.load('trained/nb-classifier.joblib')


class SVMClassifier(Classifier):

    def model_class(self):
        return joblib.load('trained/svm-classifier.joblib')


class KNNClassifier(Classifier):

    def model_class(self):
        return joblib.load('trained/knn-classifier.joblib')


class RandomForestClassifier(Classifier):
    def model_class(self):
        return joblib.load('trained/rf-classifier.joblib')


class MLPClassifier(Classifier):

    def model_class(self):
        return joblib.load('trained/mlp-classifier.joblib')
