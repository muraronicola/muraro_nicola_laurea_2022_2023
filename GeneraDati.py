import pickle
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import itertools
from scipy import signal
import os

def getMeanSignal(raw_signal):
    return np.mean(raw_signal)

def getStandardDeviation(raw_signal):
    return np.std(raw_signal)

def getMeanABS_FirstDifference(raw_signal):
    sommatoria = 0
    nSample = len(raw_signal) - 1
    for i in range(nSample):
        sommatoria = sommatoria + abs(raw_signal[i + 1] - raw_signal[i])
    result = sommatoria / (nSample)
    return result

def getMeanABS_FirstDifference_STANDARDIZED(raw_signal):
    standardized_signal = getStandardizedSignal(raw_signal)
    return getMeanABS_FirstDifference(standardized_signal)


def getMeanABS_SecondDifference(raw_signal):
    sommatoria = 0
    nSample = len(raw_signal) - 2
    for i in range(nSample):
        sommatoria = sommatoria + abs(raw_signal[i + 2] - raw_signal[i])
    result = sommatoria / (nSample)
    return result

def getMeanABS_SecondDifference_STANDARDIZED(raw_signal):
    standardized_signal = getStandardizedSignal(raw_signal)
    return getMeanABS_SecondDifference(standardized_signal)


def getStandardizedSignal(raw_signal):
    standardized_signal = np.empty(0)
    nSample = len(raw_signal)
    meanSignal = getMeanSignal(raw_signal)
    standardDeviationSignal = getStandardDeviation(raw_signal)

    for i in range(nSample):
        standardized_signal = np.append(standardized_signal, [(raw_signal[i] - meanSignal) / standardDeviationSignal], axis=0)

    return standardized_signal

def getREE(Etot, EBand):
    return EBand / Etot

def getLREE(Etot, EBand):
    return math.log(getREE(Etot,EBand),10)

def getALREE(Etot, EBand):
    return abs(getLREE(Etot, EBand))

def getEntropy(signal, allFrequencyBandPowerSpectrum):
    entropy = 0
    for i in range(len(signal)):
        prob_distr_theta = ((signal[i]) / allFrequencyBandPowerSpectrum)
        entropy += prob_distr_theta * math.log(prob_distr_theta)
    entropy = entropy * -1
    return entropy

def getEnergy(signal):
    energy = 0
    for i in range(len(signal)):
        energy += signal[i]
    return energy

def getBandFeatures(f_band, E_tot, allFrequencyBandPowerSpectrum):
    entropy = getEntropy(f_band, allFrequencyBandPowerSpectrum)
    energy_band = getEnergy(f_band)
    REE = getREE(E_tot, energy_band)
    LREE = getLREE(E_tot, energy_band)
    ALEE = getALREE(E_tot, energy_band)

    return [entropy, energy_band, REE, LREE, ALEE]


def generateFeatures(data, window_length_seconds, overlapp, selected_channel):
    data_shape = data.shape
    matrix_features = []

    lengthSegment = 128*window_length_seconds
    nWindow = int(60 // window_length_seconds)
    offset_overlapp = lengthSegment//4

    for video in range(data_shape[0]):
        for over in range(0, overlapp):
            nWLostOver = 0 if over == 0 else 1
            for window in range(1, nWindow + 1 - nWLostOver):
                start_index = int(lengthSegment * (window - 1)) + over*offset_overlapp
                end_index = int(lengthSegment * (window)) + over*offset_overlapp
                features = np.empty(0)

                for channel in selected_channel:
                    dataSegment = data[video][channel][start_index:end_index]

                    mean = getMeanSignal(dataSegment)
                    standardDeviation = getStandardDeviation(dataSegment)
                    meanASB_FD = getMeanABS_FirstDifference(dataSegment)
                    meanASB_FD_STANDARD = getMeanABS_FirstDifference_STANDARDIZED(dataSegment)
                    meanASB_SD = getMeanABS_SecondDifference(dataSegment)
                    meanASB_SD_STANDARD = getMeanABS_SecondDifference_STANDARDIZED(dataSegment)
                    features = np.append(features, [mean, standardDeviation, meanASB_FD, meanASB_FD_STANDARD, meanASB_SD, meanASB_SD_STANDARD], axis=0)

                    f, Pxx_den = signal.welch(dataSegment, fs=128)

                    Pxx_den_filtered = Pxx_den[np.logical_and(f >= 4, f <= 45)]
                    allFrequencyBandPowerSpectrum = np.sum(Pxx_den_filtered)

                    theta = Pxx_den[np.logical_and(f >= 4, f < 8)]
                    alpha = Pxx_den[np.logical_and(f >= 8, f < 13)]
                    beta  = Pxx_den[np.logical_and(f >= 13, f < 30)]
                    gamma = Pxx_den[np.logical_and(f >= 30, f <= 45)]

                    bands = [theta, alpha, beta, gamma]

                    E_tot = getEnergy(Pxx_den_filtered)

                    for f_band in bands:
                        result = getBandFeatures(f_band, E_tot, allFrequencyBandPowerSpectrum)
                        features = np.append(features, result, axis=0)
                    

                features = features.flatten()
                matrix_features.append(features)

    scaler = StandardScaler()
    matrix_features = scaler.fit_transform(matrix_features)
    return np.array(matrix_features)

def getLabelsNormalized(labels):
    newLabels = np.empty((len(labels),2))
    for i in range(len(labels)):
        valence = (labels[i][0] - 1) / 8
        arousal = (labels[i][1] - 1) / 8
        newLabels[i][0] = valence
        newLabels[i][1] = arousal

    return newLabels


def checkClass(label):
    result = False
    if ((label[0] >= 0.5 and label[1] >= 0.5) or (label[0] < 0.5 and label[1] < 0.5)):
        result = True
    return result


def get_HH_or_LL_Data(original_data, original_labels):
    data = np.empty((0, 32, 7680))
    labels = np.empty((0, 2))

    for i in range(len(original_labels)):
        if checkClass(original_labels[i]):
            data = np.append(data, [original_data[i]], axis=0)
            labels = np.append(labels, [original_labels[i]], axis=0)

    return data, labels


def findBestSplit(labels, using_SMOTE):
    nVideo = len(labels)
    nVideoTrain = math.floor(nVideo*0.8)

    videoTrain = []
    percentuale_test = 0
    percentuale_train = 0
    bestDifference = 100

    for comb in itertools.combinations(range(len(labels)), nVideoTrain):
        contatoreLabel_test = 0
        contatoreLabel_train = 0
        
        for i in range(len(labels)):
            label = 0 if labels[i][0] >= 0.5 else 1
            if i in comb:
                contatoreLabel_train = contatoreLabel_train + label
            else:
                contatoreLabel_test = contatoreLabel_test + label
                
        percentuale_test = contatoreLabel_test / (len(labels) - nVideoTrain) 
        percentuale_train = contatoreLabel_train / nVideoTrain

        if using_SMOTE:
            difference = abs(percentuale_test - 0.5)
            if difference <= bestDifference and (percentuale_train != 0 and percentuale_train != 1):
                bestDifference = difference
                videoTrain = comb
        else:
            difference = abs(percentuale_test - percentuale_train) 
            if difference <= bestDifference:
                bestDifference = difference
                videoTrain = comb

    return videoTrain


def SplitTrainTestSet(features, labels, using_SMOTE):
    features_shape = features.shape
    index_TrainSet = findBestSplit(labels, using_SMOTE)
    number_of_features_for_label = len(features) // len(labels)

    dataTrain = np.empty((0, features_shape[1])) 
    dataTest = np.empty((0, features_shape[1])) 
    labelTrain = np.empty((0, 2)) 
    labelTest = np.empty((0, 2)) 
    
    for lbl in range(len(labels)):
        if lbl in index_TrainSet:
            for i in range(lbl*number_of_features_for_label, (lbl+1)*number_of_features_for_label):
                dataTrain = np.append(dataTrain, [features[i]], axis=0)
                labelTrain = np.append(labelTrain, [labels[lbl]], axis=0)
        else:
            for i in range(lbl*number_of_features_for_label, (lbl+1)*number_of_features_for_label):
                dataTest = np.append(dataTest, [features[i]], axis=0)
                labelTest = np.append(labelTest, [labels[lbl]], axis=0)
    
    return dataTrain, dataTest, labelTrain, labelTest

def convertLabelForSMOTE(labelTrain):
    lblSmote = np.empty((labelTrain.shape[0]))

    for i in range(len(labelTrain)):
        label =  0 if labelTrain[i][0] >= 0.5 else 1
        lblSmote[i] = label

    return lblSmote

def convertLabelBackFromSMOTE(lblSmote):
    converted_label = np.empty((lblSmote.shape[0], 2))

    for i in range(len(lblSmote)):
        label = [0,0] 
        label[int(lblSmote[i])] = 1
        converted_label[i] = label

    return converted_label

def convertLabelForNN(lbl):
    converted_label = np.empty((lbl.shape[0],2))

    for i in range(len(lbl)):
        label = [0,0] 
        indexLabel = 0 if lbl[i][0] >= 0.5 else 1
        label[indexLabel] = 1
        converted_label[i] = label
    
    return converted_label


def ApplySMOTE(dataTrain, labelTrain):
    oversample = SMOTE(random_state=42)
    label_converted_forSMOTE = convertLabelForSMOTE(labelTrain)

    SMOTE_dataTrain, SMOTE_labelTrain = oversample.fit_resample(dataTrain, label_converted_forSMOTE)

    SMOTE_labelTrain = convertLabelBackFromSMOTE(SMOTE_labelTrain)
    return SMOTE_dataTrain, SMOTE_labelTrain


def Generate(dataset_path, subject_no, folder, window_length_seconds, overlapp, selected_channel, using_SMOTE):
    deap_dataset = pickle.load(open(dataset_path + subject_no + '.dat', 'rb'), encoding='latin1')

    original_data = np.array(deap_dataset['data'])
    original_data = original_data[0:40, 0:32, 384:8064]

    original_labels = np.array(deap_dataset['labels'])
    original_labels = getLabelsNormalized(original_labels)

    data, labels = get_HH_or_LL_Data(original_data, original_labels)

    features = generateFeatures(data, window_length_seconds, overlapp, selected_channel)

    dataTrain, dataTest, labelTrain, labelTest = SplitTrainTestSet(features, labels, using_SMOTE)

    if using_SMOTE:
        dataTrain, labelTrain = ApplySMOTE(dataTrain, labelTrain)

    labelTrain = convertLabelForNN(labelTrain)
    labelTest = convertLabelForNN(labelTest)

    np.savetxt(folder + 'train_data.txt', dataTrain, fmt='%1.8f')
    np.savetxt(folder + 'test_data.txt', dataTest, fmt='%1.8f')
    np.savetxt(folder + 'train_label.txt', labelTrain, fmt='%1.8f')
    np.savetxt(folder + 'test_label.txt', labelTest, fmt='%1.8f')



def main():

    dataset_path = ''

    if(dataset_path != ''):
        genData = True
        using_SMOTE = True
        window_length_seconds = 4
        subjects = range(1, 33)
        selected_channel = [0]  
        isoverlapp = True
        overlapp = 1 if isoverlapp == False else 4

        outFolder = "./DATA"
        if not os.path.exists(outFolder):
            os.mkdir(outFolder)

        if genData:
            for i in subjects:
                if(i<= 9):
                    prefisso = 's0'
                else:
                    prefisso = 's'
                subject_no = prefisso + str(i)
                folder = outFolder + "/"+ str(subject_no) + "/"

                if not os.path.exists(folder):
                    os.mkdir(folder)
                Generate(dataset_path, subject_no, folder, window_length_seconds, overlapp, selected_channel, using_SMOTE)
                print(subject_no + " completato")
    else:
        print("Errore, è necessario specificare il percorso contenente il dataset")

if __name__ == '__main__':
    main()