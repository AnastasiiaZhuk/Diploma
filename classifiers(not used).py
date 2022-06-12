from abc import ABC, abstractmethod

import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import neural_network, naive_bayes
from sklearn.ensemble import RandomForestClassifier as RandomForestClassifier_


class Classifier(ABC):
    @abstractmethod
    def model_class(self):
        pass


class NBClassifier(Classifier):

    def model_class(self):
        model = naive_bayes.GaussianNB()
        data = pd.DataFrame([20], ['max_iter'])
        return model, data


class MLPClassifier(Classifier):

    def model_class(self):
        alpha = st.sidebar.number_input("Alpha", 0.01, 10.0, step=0.01, key='alpha')
        max_iter = st.sidebar.slider("Maximum number of iterations", 200, 1000, key='max_iter')

        model = neural_network.MLPClassifier(alpha=alpha, max_iter=max_iter)
        data = pd.DataFrame([alpha, max_iter], ['alpha', 'max_iter'])
        return model, data


class RandomForestClassifier(Classifier):
    def model_class(self):
        n_estimators = st.sidebar.number_input(
                "The number of trees in the forest",
                100, 5000, step=10, key='new'
            )
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1)
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'))
        n_jobs = -1
        model = RandomForestClassifier_(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=n_jobs)
        data = pd.DataFrame([n_estimators, max_depth, bootstrap, n_jobs], ['n_estimators', 'max_depth', 'bootstrap', 'n_jobs'])
        return model, data


class SVMClassifier(Classifier):

    def model_class(self):
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_SVM')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')
        model = SVC(C=C, kernel=kernel, gamma=gamma)
        data = pd.DataFrame([C, kernel, gamma], ['C', 'gamma', 'kernel'])
        return model, data


class KNNClassifier(Classifier):

    def model_class(self):
        n_neighbors = st.sidebar.number_input("Num. of neighbors", 1, 100, step=1, key='n_neighbors')
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        data = pd.DataFrame([n_neighbors], ['n_neighbors'])
        return model, data

