import random
import mne
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
import torch
import torch.nn as nn

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score
from scipy.stats import sem

warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')
random.seed(10)
np.random.seed(10)

def CaricaDati(startFolder):
    vector_data_train = np.loadtxt(startFolder+ 'train_data.txt', dtype=float)
    vector_four_class_labels_train = np.loadtxt(startFolder + 'train_label.txt', dtype=float)
    vector_data_test = np.loadtxt(startFolder + 'test_data.txt', dtype=float)
    vector_four_class_labels_test = np.loadtxt(startFolder + 'test_label.txt', dtype=float)
    return vector_data_train, vector_four_class_labels_train, vector_data_test, vector_four_class_labels_test


def macro_double_soft_f1(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y_hat = torch.sigmoid(y_hat)
    tp = torch.sum(y_hat * y, axis=1)
    fp = torch.sum(y_hat * (1 - y), axis=1)
    fn = torch.sum((1 - y_hat) * y, axis=1)
    tn = torch.sum((1 - y_hat) * (1 - y), axis=1)
    soft_f1_class1 = 2*tp / (2*tp + fn + fp + 1e-16)
    cost_class1 = (1 - soft_f1_class1)
    
    return torch.mean(cost_class1)


def Train_LDA(train_data, train_label, test_data):
    clf = LinearDiscriminantAnalysis()
    clf.fit(train_data, train_label)

    prediction = clf.predict(test_data)
    return prediction


def Train_RF(train_data, train_label, test_data):
    clf = RandomForestClassifier(max_depth=1, random_state=0)
    clf.fit(train_data, train_label)

    prediction = clf.predict(test_data)
    return prediction


def Train_SVM(train_data, train_label, test_data):
    clf = SVC(kernel='rbf')
    clf.fit(train_data, train_label)

    prediction = clf.predict(test_data)
    return prediction


def Train_XGB(train_data, train_label, test_data):
    clf = GradientBoostingClassifier()
    clf.fit(train_data, train_label)

    prediction = clf.predict(test_data)
    return prediction

def Train_NN(train_data, train_label, test_data, test_label):
    torch.manual_seed(10)

    learning_rate = 0.01
    epocs = 5000

    nodi = 50
    model = nn.Sequential(
        nn.Linear(len(train_data[0]), nodi),
        nn.ELU(),
        nn.Linear(nodi, nodi),
        nn.ELU(),
        nn.Linear(nodi, 2)
    )

    x = torch.tensor(train_data)
    x = x.to(torch.float32)

    test_x = torch.tensor(test_data)
    test_x = test_x.to(torch.float32)

    y = torch.tensor(train_label)
    y = y.to(torch.float32)

    test_y = torch.tensor(test_label)
    test_y = test_y.to(torch.float32)

    tot_pos_label = 0
    for i in range(len(train_label)):
        tot_pos_label = tot_pos_label + train_label[i][1]

    pos_weight = tot_pos_label / (len(train_label) - tot_pos_label)

    bce_loss = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor(pos_weight))
    soft_f1 = macro_double_soft_f1
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    bce_importance = 0.3
    soft_f1_importance = 0.7

    for t in range(epocs):
        model.train()
        y_pred = model(x)
        loss = bce_importance*bce_loss(y_pred, y) + soft_f1_importance*soft_f1(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred_test = model(test_x)
        torch.set_printoptions(threshold=10_000)

    convertedPred = np.empty(0)

    for pred in y_pred_test:
        if pred[0] >= pred[1]:
            convertedPred = np.append(convertedPred, 0)
        else:
            convertedPred = np.append(convertedPred, 1)

    return convertedPred

def ConvertLabelToNumbers(labels):
    new_label = np.zeros((len(labels)))
    for i in range(len(labels)):
        if labels[i][0] == 0:
            new_label[i] = 1

    return new_label


def TrainOneSubject(pathSingleUser, method):
    data_train, four_class_label_train,  data_test, four_class_labels_test = CaricaDati(pathSingleUser)

    if method != 4:
        four_class_label_train = ConvertLabelToNumbers(four_class_label_train)
        four_class_labels_test = ConvertLabelToNumbers(four_class_labels_test)

    if method == 0:
        prediction = Train_LDA(data_train, four_class_label_train, data_test)
    elif method == 1:
        prediction = Train_RF(data_train, four_class_label_train, data_test)
    elif method == 2:
        prediction = Train_SVM(data_train, four_class_label_train, data_test)
    elif method == 3:
        prediction = Train_XGB(data_train, four_class_label_train, data_test)
    elif method == 4:
        prediction = Train_NN(data_train, four_class_label_train, data_test, four_class_labels_test)
        four_class_labels_test = ConvertLabelToNumbers(four_class_labels_test)

    recall = recall_score(prediction, four_class_labels_test)
    precision = precision_score(prediction, four_class_labels_test)
    f1 = f1_score(prediction, four_class_labels_test)
    balanced_acc = balanced_accuracy_score(prediction, four_class_labels_test)

    return recall*100, precision*100, f1*100, balanced_acc*100


def main():
    startFolder = "./DATA_RIDOTTI"
    titolo = "Miglior configurazione per la classificazione binaria"

    font = {'family' : 'normal',
        'size' : 20}

    plt.rc('font', **font)
    selected_subject = range(1,33)
    methods = [0, 1, 2, 3, 4]

    vector_recall_vector = np.empty((0, len(selected_subject)))
    vector_precision_vector = np.empty((0, len(selected_subject)))
    vector_f1_vector = np.empty((0, len(selected_subject)))
    vector_balancedAcc_vector = np.empty((0, len(selected_subject)))

    str_models = ["LDA", "RF", "SVM", "XBG", "NN"]
    for method in methods:

        recall_vector = np.zeros(len(selected_subject))
        precision_vector = np.zeros(len(selected_subject))
        f1_vector = np.zeros(len(selected_subject))
        balancedAcc_vector = np.zeros(len(selected_subject))

        for SubjectNumber in selected_subject:

            if(SubjectNumber<=9):
                prefisso = 's0'
            else:
                prefisso = 's'

            pathSingleUser = startFolder + "/" + prefisso + str(SubjectNumber) + "/"
            recall, precision, f1, balanced_acc = TrainOneSubject(pathSingleUser, method)

            recall_vector[SubjectNumber - 1] = recall
            precision_vector[SubjectNumber - 1] = precision
            f1_vector[SubjectNumber - 1] = f1
            balancedAcc_vector[SubjectNumber - 1] = balanced_acc

        recall_tot = np.mean(recall_vector, axis=0)
        precision_tot = np.mean(precision_vector, axis=0)
        f1_tot = np.mean(f1_vector, axis=0)
        balancedAcc_tot = np.mean(balancedAcc_vector, axis=0)

        vector_recall_vector = np.append(vector_recall_vector, [recall_vector], axis=0)
        vector_precision_vector = np.append(vector_precision_vector, [precision_vector], axis=0)
        vector_f1_vector = np.append(vector_f1_vector, [f1_vector], axis=0)
        vector_balancedAcc_vector = np.append(vector_balancedAcc_vector, [balancedAcc_vector], axis=0)
        print("\n\n" + str_models[method] + " - Risultato complessivo")
        print("Accuratezza bilanciata: " + str(round(balancedAcc_tot, 2)))
        print("Richiamo: " + str(round(recall_tot, 2)))
        print("Precisione: " + str(round(precision_tot, 2)))
        print("F1 score: " + str(round(f1_tot, 2)))


    fig, ax = plt.subplots(layout='constrained')
    width = 0.22
    multiplier = 0
    nMethods = len(methods)
    x = np.arange(nMethods)
    nMetriche = 4

    plt_balancedAcc_vector = [np.mean(vector_balancedAcc_vector, axis=1), sem(vector_balancedAcc_vector, axis=1)]
    plt_vector_recall_vector = [np.mean(vector_recall_vector, axis=1), sem(vector_recall_vector, axis=1)]
    plt_vector_precision_vector = [np.mean(vector_precision_vector, axis=1), sem(vector_precision_vector, axis=1)]
    plt_vector_f1_vector = [np.mean(vector_f1_vector, axis=1), sem(vector_f1_vector, axis=1)]

    measurement = [plt_balancedAcc_vector, plt_vector_recall_vector, plt_vector_precision_vector, plt_vector_f1_vector]
    attribute = ["Accuratezza bilanciata", "Richiamo", "Precisione", "F1 score"]

    for i in range(nMetriche):
        offset = width * multiplier
        rects = ax.bar(x + offset, np.round(measurement[i][0], decimals=1) , width, label=attribute[i],  yerr=measurement[i][1], capsize=10)

        ax.errorbar
        ax.bar_label(rects, padding=nMetriche)
        multiplier += 1

    ax.set_ylabel('Percentuale')
    ax.set_xlabel('Modelli')
    ax.set_title(titolo)
    ax.set_xticks(x + 0.33, str_models)
    ax.legend(loc='upper left', ncols=nMetriche)
    ax.set_ylim(20, 70)

    plt.show()


if __name__ == '__main__':
    main()