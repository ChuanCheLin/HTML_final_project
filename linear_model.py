from re import split
import re
from liblinear.liblinearutil import train
from liblinear.liblinearutil import predict
import numpy as np
from liblinear.liblinearutil import *
import csv
from numpy.core.numeric import NaN
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Normalizer

# read training & testing data
def data_loader(train_path, test_path):
    with open(train_path, 'r') as fp:     
        data_train = list(csv.reader(fp))
        train_id = np.array(data_train[1:])[:, :1]
        data_train = np.array(data_train[1:])[:, 1:]
    
    with open(test_path, 'r') as fp:     
        data_test = list(csv.reader(fp))
        test_id = np.array(data_test[1:])
        data_test = np.array(data_test[1:])[:, 1:]
    
    return data_train, train_id, data_test, test_id

# encode y, which is our objective 'Churn Category'
def encode_y(data):
    from sklearn.preprocessing import OrdinalEncoder
    enc = OrdinalEncoder(categories=[['No Churn', 'Competitor', 'Dissatisfaction', 'Attitude', 'Price', 'Other']])
    y = enc.fit_transform(data[:, 44:45])
    for i in range((len(y))):
        y[i] = int(y[i])
    return y

# extract numerical features that we want to use for training
# ex: [2] => Age, [15] => Satisfication_Score, ...
def feature_extractor(data_trainval, data_test, feats):
    x = data_trainval[:, feats]
    x = (x.astype(np.float))
    x = np.nan_to_num(x)

    x_test = data_test[:, feats]
    x_test = (x_test.astype(np.float))
    x_test = np.nan_to_num(x_test)
    return x, x_test

def normalize(data_trainval, data_test):
    scaler = Normalizer().fit(data_trainval)
    data_trainval = scaler.transform(data_trainval)
    scaler = Normalizer().fit(data_test)
    data_test = scaler.transform(data_test)
    return data_trainval, data_test

# output prediction 
def make_pred(pred, test_id):
    with open('pred.csv', 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['Customer ID', 'Churn Category'])
        for i, p in enumerate(pred): 
            writer.writerow([test_id[i, 0], int(p)])

def zero_one_err(y_gt, y_predict):  
    zero_one_err = 0
    assert len(y_gt) == len(y_predict)
    N = len(y_gt)
    for i in range(N):
        if y_gt[i] != y_predict[i]:
            zero_one_err += 1
    zero_one_err = (zero_one_err/N)
    return zero_one_err

def V_folder_cross_validation(x_train, y_train, param, V): #V-folder
    assert len(x_train) == len(y_train)
    best_param = None
    best_err = 1
    N = len(x_train)
    n = N/V
    P = len(param)
    Err_cv_tot = []
    for j in range(P):
        Err_cv = 0
        for i in range(V):
            x_val = x_train[int(i*n):int((i+1)*n),:]
            y_val = y_train[int(i*n):int((i+1)*n)]
            x_train_minus = np.delete(x_train, range(int(i*n),int((i+1)*n)), axis=0)
            y_train_minus = np.delete(y_train, range(int(i*n),int((i+1)*n)), axis=0)
            assert len(x_train_minus) == len(y_train_minus)
            model_minus = train(y_train_minus, x_train_minus, param[j])
            p_label, p_acc, p_val = predict(y_val, x_val, model_minus)# validation
            Err_val = zero_one_err(y_val, p_label)
            Err_cv += (Err_val)
        Err_cv = Err_cv/V
        Err_cv_tot.append(Err_cv)
        if Err_cv < best_err:
            best_err = Err_cv
            best_param = j 
    print(Err_cv_tot)
    return best_param

data_trainval, train_id, data_test, test_id = data_loader('trainval_data.csv','Test_data.csv')
y = np.ravel(encode_y(data_trainval))

# 2-Age, 7-Number_of_Dependents, 15-Satisfication_Score, 19-Tenure_in_Months, 38-Monthly_Charge, 39-Total_Charges
x_trainval, x_test = feature_extractor(data_trainval, data_test, [2, 7, 15, 19, 38, 39])

# Split data
X_train, X_val, y_train, y_val = train_test_split(x_trainval, y, random_state = 0)

#-s 0 => regularized logistic regression 
param = [               
        '-s 0 -c 10000 -e 0.000001',
        '-s 0 -c 100 -e 0.000001',
        '-s 0 -c 1 -e 0.000001',
        '-s 0 -c 0.01 -e 0.000001',
        '-s 0 -c 0.0001 -e 0.000001'
]

best_param_id = V_folder_cross_validation(x_trainval, y, param, 10)
best_model = train(y_train, X_train, param[best_param_id])
predict(y_val, X_val, best_model)
test_label, p_acc, p_val = predict([], x_test, best_model)

make_pred(test_label, test_id)