from re import split
import numpy as np
import csv
from numpy.core.numeric import NaN
from sklearn.model_selection import train_test_split
#read training & testing data
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

#encode y, which is our objective 'Churn Category'
def encode_y(data):
    from sklearn.preprocessing import OrdinalEncoder
    enc = OrdinalEncoder(categories=[['No Churn', 'Competitor', 'Dissatisfaction', 'Attitude', 'Price', 'Other']])
    y = enc.fit_transform(data[:, 44:45])
    for i in range((len(y))):
        y[i] = int(y[i])
    return y

# extract numerical features that we want to use for training
# ex: [2] => Age, [15] => Satisfication_Score, ...
def feature_extractor(data_train, data_test, feats):
    x = data_train[:, feats]
    x = (x.astype(np.float))
    x = np.nan_to_num(x)

    x_test = data_test[:, feats]
    x_test = (x_test.astype(np.float))
    x_test = np.nan_to_num(x_test)
    return x, x_test

# Naive Bayes classifier Training
def NB(X_train, X_val, y_train, y_val):
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import confusion_matrix
    gnb = GaussianNB().fit(X_train, y_train)
    val_predictions = gnb.predict(X_val)

    # Validation accuracy
    accuracy = gnb.score(X_val, y_val)
    print(accuracy)
    # Validation confusion matrix
    cm = confusion_matrix(y_val, val_predictions)
    print(cm)

    #testing
    test_predictions = gnb.predict(x_test)
    return test_predictions

#output prediction 
def make_pred(pred):
    with open('pred.csv', 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['Customer ID', 'Churn Category'])
        for i, p in enumerate(pred): 
            writer.writerow([test_id[i, 0], int(p)])

data_trainval, train_id, data_test, test_id = data_loader('trainval_data.csv','Test_data.csv')
y = encode_y(data_trainval)
x_trainval, x_test = feature_extractor(data_trainval, data_test, [2, 15])

# Split data
X_train, X_val, y_train, y_val = train_test_split(x_trainval, y, random_state = 0)

predictions_NB = NB(X_train, X_val, y_train, y_val)

make_pred(predictions_NB)