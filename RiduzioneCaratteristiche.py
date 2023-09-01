import random
import mne
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from captum.attr import FeaturePermutation
import os

warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')
random.seed(10)
np.random.seed(10)


def getFeatureImportance(data_train):
    torch.manual_seed(10)

    nodi = 50
    model = nn.Sequential(
        nn.Linear(len(data_train[0]), nodi),
        nn.ELU(),
        nn.Linear(nodi, nodi),
        nn.ELU(),
        nn.Linear(nodi, 2)
    )

    x = torch.tensor(data_train)
    x = x.to(torch.float32)

    feature_perm = FeaturePermutation(model)
    attr = feature_perm.attribute(x, target=1)

    meanValue_subject = np.mean(np.array(attr), axis=0)
    return meanValue_subject


def CaricaDati(startFolder):
    vector_data_train = np.loadtxt(startFolder + 'train_data.txt', dtype=float)
    vector_four_class_labels_train = np.loadtxt(
        startFolder + 'train_label.txt', dtype=float)
    vector_data_test = np.loadtxt(startFolder + 'test_data.txt', dtype=float)
    vector_four_class_labels_test = np.loadtxt(
        startFolder + 'test_label.txt', dtype=float)
    return vector_data_train, vector_four_class_labels_train, vector_data_test, vector_four_class_labels_test


def GetFeatureImportanceOneSubject(SubjectNumber, pathSingleUser):
    data_train, _,  _, _ = CaricaDati(pathSingleUser)
    fatureImportante = getFeatureImportance(data_train)
    return fatureImportante



def ReduceFeatures(dati, keptFeatures, nKeptfeatures):
    features_reduced = np.zeros((len(dati), nKeptfeatures))

    for i in range(len(dati)):
        features = np.empty(0)
        for j in range(len(dati[0])):
            if keptFeatures[j] == 1:
                features = np.append(features, [dati[i][j]], axis=0)
        features_reduced[i] = features

    return features_reduced
    

def GenerateDataFeatureAnalysies(sourcePath, destPath, keptFeatures, nKeptfeatures):
    data_train, label_train,  data_test, label_test = CaricaDati(sourcePath)
    data_train = ReduceFeatures(data_train, keptFeatures, nKeptfeatures)
    data_test = ReduceFeatures(data_test, keptFeatures, nKeptfeatures)

    np.savetxt(destPath + 'train_data.txt', data_train, fmt='%1.8f')
    np.savetxt(destPath + 'test_data.txt', data_test, fmt='%1.8f')
    np.savetxt(destPath + 'train_label.txt', label_train, fmt='%1.8f')
    np.savetxt(destPath + 'test_label.txt', label_test, fmt='%1.8f')


def main():
    startFolder = "./DATA"
    outFolder = "./DATA_RIDOTTI"

    if not os.path.exists(outFolder):
        os.mkdir(outFolder)

    selected_subject = range(1, 33)
    nFeatures = 26
    meanValue_array = np.zeros((len(selected_subject), nFeatures))

    for SubjectNumber in selected_subject:
        if(SubjectNumber<=9):
            prefisso = 's0'
        else:
            prefisso = 's'

        pathSingleUser = startFolder + "/" + prefisso + str(SubjectNumber) + "/"
        meanValue = GetFeatureImportanceOneSubject(SubjectNumber, pathSingleUser)
        meanValue_array[SubjectNumber - 1] = meanValue

    meanValue_array = np.mean(np.array(meanValue_array), axis=0)

    keptFeatures = np.zeros(nFeatures)
    nKeptfeatures = 0
    for i in range(nFeatures):
        if meanValue_array[i] > 0:
            keptFeatures[i] = 1
            nKeptfeatures = nKeptfeatures + 1

    for SubjectNumber in selected_subject:

        if(SubjectNumber<=9):
            prefisso = 's0'
        else:
            prefisso = 's'

        sourcePath = startFolder + "/" + prefisso + str(SubjectNumber) + "/"
        destPath = outFolder + "/" + prefisso + str(SubjectNumber)

        if not os.path.exists(destPath):
            os.mkdir(destPath)
        destPath = destPath + "/"

        GenerateDataFeatureAnalysies(sourcePath, destPath, keptFeatures, nKeptfeatures)
        print(str(SubjectNumber) + " completato")


if __name__ == '__main__':
    main()
