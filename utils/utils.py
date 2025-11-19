import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, classification_report, \
    roc_curve, auc, precision_score
from sklearn.model_selection import LeaveOneOut

from joblib import Parallel, delayed
from CrabNet_eg.kingcrab import CrabNet
from CrabNet_eg.model import Model
from CrabNet_eg.get_compute_device import get_compute_device
# No warnings about setting value on copy of slice
pd.options.mode.chained_assignment = None

# Display up to 60 columns of a dataframe
pd.set_option('display.max_columns', 60)

# Matplotlib visualization
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
# Internal ipython tool for setting figure size
import warnings

warnings.filterwarnings('ignore')

config = {
    "mathtext.fontset": 'stix',
    "font.family": 'serif',
    "font.serif": ['Times New Roman'],
    "font.size": 24,
    'axes.unicode_minus': False
}
rcParams.update(config)
plt.rcParams['axes.unicode_minus'] = False
large = 22
med = 16
small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (8, 6),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.rcParams['figure.dpi'] = 300
seed = 42
import os
from mp_api.client import MPRester


def plot_binary_confusion_matrix(y_true, y_pred, classes=['Indirect', 'Direct'], normalize=False,
                                 title='Confusion Matrix', cmap=plt.cm.GnBu):
    """
    绘制二分类混淆矩阵

    参数:
    y_true (array-like): 真实标签（一维数组或列表）
    y_pred (array-like): 预测标签（一维数组或列表）
    classes (list): 类别名称列表，默认为['Negative', 'Positive']
    normalize (bool): 是否对混淆矩阵进行归一化，默认为False
    title (str): 图像标题，默认为'Confusion Matrix'
    cmap (matplotlib colormap): 颜色图谱，默认为plt.cm.Blues
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def loo(model, X, y, name=""):
    leave_one_out = LeaveOneOut()
    tests = []
    preds = []
    try:
        X = X.values
        y = y.values
    except:
        pass

    # 使用留一验证法进行交叉验证
    def train_and_predict(train_index, test_index):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 训练模型
        model.fit(X_train, y_train)

        # 在测试集上进行预测
        y_pred = model.predict(X_test)
        return y_test, y_pred

    results = Parallel(n_jobs=24)(
        delayed(train_and_predict)(train_index, test_index) for train_index, test_index in leave_one_out.split(X)
    )

    for result in results:
        y_test, y_pred = result
        tests.append(y_test)
        preds.append(y_pred)

    accuracy = accuracy_score(tests, preds)
    f1 = f1_score(tests, preds)
    mcc = matthews_corrcoef(tests, preds)
    report = classification_report(tests, preds)
    recall = float(report.split()[11])

    return {'accuracy': accuracy, 'f1-score': f1, 'mcc': mcc, 'recall': recall, 'cs': 0.5*f1+0.5*accuracy}


def foldn_score(model, X, y,n):
    kf = StratifiedKFold(n_splits=n, shuffle=True, random_state=42)
    accuracies = []
    f1s = []
    mccs = []
    recalls = []
    best_score = 0
    best_model = None
    train_data = []

    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        try:
            X = X.values
            y = y.values
        except:
            pass
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred))
        mccs.append(matthews_corrcoef(y_test, y_pred))
        report = classification_report(y_test, y_pred)
        recall = float(report.split()[11])
        recalls.append(recall)
        cs = 0.5*accuracy_score(y_test, y_pred) + 0.5*f1_score(y_test, y_pred)
        if cs > best_score:
            best_score = cs
            best_model = model
            train_data = [X_train,y_train]
    result = {
        'accuracy': np.mean(accuracies),
        'f1-score': np.mean(f1s),
        'mcc': np.mean(mccs),
        'recall': np.mean(recalls),
        'cs': 0.5*np.mean(accuracies)+0.5*np.mean(f1s),
    }
    return result,best_model,train_data


def loos(models, X, y, names=[]):
    leave_one_out = LeaveOneOut()
    result_df = []
    for model, name in zip(models, names):
        tests = []
        preds = []
        try:
            X = X.values
            y = y.values
        except:
            pass

        # 使用留一验证法进行交叉验证
        def train_and_predict(train_index, test_index):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # 训练模型
            model.fit(X_train, y_train)

            # 在测试集上进行预测
            y_pred = model.predict(X_test)
            return y_test, y_pred

        results = Parallel(n_jobs=25)(
            delayed(train_and_predict)(train_index, test_index) for train_index, test_index in leave_one_out.split(X)
        )

        for result in results:
            y_test, y_pred = result
            tests.append(y_test)
            preds.append(y_pred)

        accuracy = accuracy_score(tests, preds)
        f1 = f1_score(tests, preds)
        mcc = matthews_corrcoef(tests, preds)
        report = classification_report(tests, preds)
        recall = float(report.split()[11])

        precision = precision_score(tests, preds)
        result_df.append({'model': name, 'accuracy': accuracy, 'f1-score': f1, 'mcc': mcc, 'recall': recall,
                          'cs': 0.5*accuracy+0.5*f1})

    return pd.DataFrame(result_df)


def foldns(models, X, y, n,names=[]):
    result_df = []
    best_score = 0
    best_model = None
    train_data = []
    for model, name in zip(models, names):
        kf = StratifiedKFold(n_splits=n, shuffle=True, random_state=42)
        accuracies = []
        f1s = []
        mccs = []
        recalls = []
        for i, (train_index, test_index) in enumerate(kf.split(X, y)):
            try:
                X = X.values
                y = y.values
            except:
                pass
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracies.append(accuracy_score(y_test, y_pred))
            f1s.append(f1_score(y_test, y_pred))
            mccs.append(matthews_corrcoef(y_test, y_pred))
            report = classification_report(y_test, y_pred)
            recall = float(report.split()[11])
            recalls.append(recall)
            cs = 0.5*accuracy_score(y_test, y_pred) + 0.5*f1_score(y_test, y_pred)
            if cs > best_score:
                best_score = cs
                best_model = model
                train_data = [X_train,y_train]
        result = {
            'model': name,
            'accuracy': np.mean(accuracies),
            'f1-score': np.mean(f1s),
            'mcc': np.mean(mccs),
            'recall': np.mean(recalls),
            'cs': 0.5*np.mean(accuracies)+0.5*np.mean(f1s),
        }
        result_df.append(result)
    return pd.DataFrame(result_df),best_model,train_data

import pickle
def save_params_to_file(dict_obj, file_path):
    with open(file_path, 'wb') as fp:
        pickle.dump(dict_obj, fp)

def load_params_from_file(file_path):
    with open(file_path, 'rb') as fp:
        dict_obj = pickle.load(fp)
    return dict_obj

    
compute_device = get_compute_device()

class Modifier:
    def __init__(self):
        self.model = Model(CrabNet(compute_device=compute_device).to(compute_device),
                           model_name='predict_new_data', verbose=True, classification=False)
        self.model.load_network(f'./models/trained_models/fold_2.pth')

    def modify(self, formula: list, band_gap: list):
        df = pd.DataFrame({'formula': formula, 'band gap': band_gap})
        df['target'] = 0
        self.model.load_data(df)
        pred = self.model.predict(self.model.data_loader)[1]
        pred = np.nan_to_num(pred, copy=True, nan=0.0)
        df['target'] = pred
        df = df.rename(columns={'target': 'hse'})
        return df