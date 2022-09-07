import librosa
import matplotlib.pyplot as plt
from glob import glob
from librosa import feature
from sklearn.linear_model import LogisticRegression
import numpy as np
import csv
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

import numpy as np
from sklearn import preprocessing
from sklearn import utils
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#directories of normal audios
# Press the green button in the gutter to run the script.



fn_list_i = [
    feature.chroma_stft,
    feature.spectral_centroid,
    feature.spectral_bandwidth,
    feature.chroma_cens,
    feature.mfcc,
    feature.melspectrogram,
    feature.spectral_contrast,
    feature.tonnetz,
    feature.chroma_cqt
]

fn_list_ii = [
    feature.rms,
    feature.zero_crossing_rate
]

def PCA_ALGO():
    names = [
        "chroma_stft","spectral_centroid","spectral_bandwidth","spectral_rolloff","rmse"
    ]
    dataset= pd.read_csv("normals_00.csv",names=names)
    # url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    # names = [sepal-length,sepal-width,petal-length,petal-width,Class]
    # dataset = pd.read_csv(url,names=names)
    print(dataset.head())
    # X = dataset.drop(Class,1)
    # y = dataset[Class]
    X = dataset.drop("spectral_bandwidth",1)
    y = dataset["spectral_bandwidth"]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    pca = PCA()
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    explained_variance = pca.explained_variance_ratio_
    pca = PCA(n_components=1)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    lab = preprocessing.LabelEncoder()
    y_transformed = lab.fit_transform(y)

    # view transformed values
    print(y_transformed)
    classifier =  linear_model.SGDRegressor()
    classifier.fit(X_train,y_train)
    # Predicting the Test set results

    y_pred = classifier.predict(X_test)

    cutoff = 1.7  # decide on a cutoff limit
    y_pred_classes = np.zeros_like(y_pred)  # initialise a matrix full with zeros
    y_pred_classes[y_pred > cutoff] = 1

    y_test_classes = np.zeros_like(y_pred)
    y_test_classes[y_test > cutoff] = 1

    cm = confusion_matrix(y_test_classes, y_pred_classes)
    print(cm)
    print("Accuracy" + accuracy_score(y_test_classes,y_pred_classes))
    # pca = PCA(n_components=2)
    # X_train = pca.fit_transform(X_train)
    # X_test = pca.transform(X_test)
    #
def get_feature_vector(y,sr):
    feat_vect_i = [np.mean(funct(y,sr)) for funct in fn_list_i]
    feat_vect_ii = [np.mean(funct(y)) for funct in fn_list_ii]
    feature_vector = feat_vect_i + feat_vect_ii
    return feature_vector


if __name__ == "__main__":
    # PCA_ALGO()
    norm_audio_list=[]
    for i in range(0,24):
        actor_dir="Actor_0"+str(i)
        if(i>=11):
            actor_dir="Actor_"+str(i)
        norm_data_dir = "C:\\Users\\Admin\\PycharmProjects\\ML_CT1\\venv\\files\\{a}\\".format(a=actor_dir)
        norm_audio_files = glob(norm_data_dir +"*.wav")
        norm_audios_feat = []
        for file in norm_audio_files:
            y,sr = librosa.load(file,sr=None)
            feature_vector = get_feature_vector(y,sr)
            norm_audios_feat.append(feature_vector)
        norm_audio_list.append(norm_audios_feat)
    print(norm_audio_list)

    norm_output = "normals_00.csv"
    header = [
    "chroma_stft",
    "spectral_centroid",
    "spectral_bandwidth",
    "chroma_cens",
    "chroma_mfcc",
    "melspectrogram",
    "spectral_contrast",
    "tonnetz",
    "chroma_cqt",
    "rmse",
    "zero_crossing_rate"
    ]

    with open(norm_output,"+w") as f:
        csv_writer = csv.writer(f,delimiter=',' )
        csv_writer.writerow(header)
        csv_writer.writerows(norm_audio_list)


