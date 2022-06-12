import os

import pystegano
import cv2
import stegano.lsb
from PIL import Image
from numpy import std, ptp, median
from scipy.stats import skew, kurtosis, mstats
import eeglib
from sklearn import preprocessing
import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score

from load_classifier import KNNClassifier, RandomForestClassifier, SVMClassifier, NBClassifier, MLPClassifier


@st.cache(persist=True)
def load_data():
    training_data = pd.read_csv(
        'train_test_datasets/features_train_70000.csv')
    testing_data = pd.read_csv(
        'train_test_datasets/features_test_70000.csv')

    return training_data, testing_data


@st.cache(persist=True)
def split(train: pd.DataFrame, test: pd.DataFrame):
    x_train, y_train = train.drop([' Tag'], axis=1), train[' Tag']
    X_test, Y_test = test.drop([' Tag'], axis=1), test[' Tag']

    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train, X_test = scaler.fit_transform(x_train), scaler.fit_transform(
        X_test)
    return x_train, y_train, X_test, Y_test


class_names = ['clean', 'stego']
x_train, y_train, X_test, Y_test = split(*load_data())


def plot_metrics(metrics_list, model):
    if 'Confusion Matrix' in metrics_list:
        st.subheader("Confusion Matrix")
        plot_confusion_matrix(model, X_test, Y_test, display_labels=class_names)
        st.pyplot()

    if 'ROC Curve' in metrics_list:
        st.subheader("ROC Curve")
        plot_roc_curve(model, X_test, Y_test)
        st.pyplot()

    if 'Precision-Recall Curve' in metrics_list:
        st.subheader('Precision-Recall Curve')
        plot_precision_recall_curve(model, X_test, Y_test)
        st.pyplot()


def geo_mean(x):
    x_new = [i for i in x if i != 0]
    a = np.log(x_new)
    return np.exp(a.mean())


def image_to_test_data(img_path):
    choose_classifier = {
        "Random Forest": RandomForestClassifier,
        "KNN Classifier": KNNClassifier,
        "Support Vector Machine (SVM)": SVMClassifier,
        'Naive Bayes Classifier': NBClassifier,
        'MLP Classifier': MLPClassifier,
    }
    vals = img_path.mean(axis=2).flatten()
    b, bins, patches = plt.hist(vals, 255)  # histogram 

    data = {'Kurtosis': [kurtosis(b)],
            ' Skewness': [skew(b)],
            ' Std': [std(b)],
            ' Range': [ptp(b)],
            ' Median': [median(b)],
            ' Geometric_Mean': [geo_mean(b)],
            ' Mobility': [eeglib.features.hjorthMobility(b)],
            ' Complexity': [eeglib.features.hjorthComplexity(b)]
            }
    df = pd.DataFrame(data)
    testing_data = pd.read_csv('train_test_datasets/features_test_70000.csv')

    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

    X_test, Y_test = testing_data.drop([' Tag'], axis=1), testing_data[' Tag']
    X_test = X_test.append(df, ignore_index=True)
    X_test = scaler.fit_transform(X_test)
    return X_test, Y_test, 'Random Forest', choose_classifier['Random Forest']().model_class().predict([X_test[13999]])


def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)


def main():
    st.title("Steganography Detect Web App")
    st.sidebar.title("Binary Steganography Detect Web App")
    st.markdown("Is your image clear or stego? ")
    st.sidebar.markdown("Is your image clear or stego? ")
    # st.subheader("Choose Classifier")
    #
    # classifiers = (
    #     "Support Vector Machine (SVM)", "Naive Bayes Classifier", "Random Forest", "MLP Classifier", "KNN Classifier")
    # classifier = st.selectbox("Classifier", classifiers)
    filename = 'image.jpg'
    uploaded_file = st.file_uploader("Choose a image file", type="jpg")

    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        st.image(opencv_image, width=200, channels='BGR')

    uploaded_file_to_stego = st.sidebar.file_uploader("Choose a image file to add LSB steganography", type="jpg")
    if uploaded_file_to_stego is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file_to_stego.read()), dtype=np.uint8)
        opencv_image1 = cv2.imdecode(file_bytes, 1)
        st.sidebar.image(opencv_image1, width=100, channels='BGR', caption='Image to be changed')
        desired_text = st.sidebar.text_input(label='Secret message to input', max_chars=100)
        desired_name_of_file = st.sidebar.text_input(label='Name of file', max_chars=20)

        if st.sidebar.button("Add steganography", key='steganography1'):
            print(desired_text)
            encoded_image = pystegano.lsb.encode(opencv_image1, desired_text)
            cv2.imwrite('./' + f'{desired_name_of_file}.jpg',
                        encoded_image)
            st.sidebar.write(
                f"Saved to: ./{desired_name_of_file}.jpg")
    #
    # class_factory = choose_classifier[classifier]
    # model = class_factory().model_class()

    if st.button("Classify", key='classify1'):

        print('-------------------------------------------')
        print(filename)
        X_test, Y_test, class_name, predicted_photo = image_to_test_data(opencv_image)
        st.subheader(f'{class_name} Results: ')
        # accuracy = model.score(X_test, Y_test)
        # y_pred = model.predict(X_test)
        # st.write("Accuracy: ", accuracy.round(2))
        # st.write("Precision: ", precision_score(Y_test, y_pred, labels=class_names).round(2))
        # st.write("Recall: ", recall_score(Y_test, y_pred, labels=class_names).round(2))
        st.write("Stego or clean? Image is: ", 'Clean' if predicted_photo == 0 else 'With Steganography')



if __name__ == '__main__':
    main()
