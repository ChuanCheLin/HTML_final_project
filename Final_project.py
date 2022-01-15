from re import X
import numpy as np
import csv
from numpy.core.defchararray import encode
from numpy.core.numeric import NaN
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,  f1_score
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

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

#encode other non-number features
def encode_other(data, data_test, feats, y):
    from category_encoders.target_encoder import TargetEncoder

    for i in feats:
        enc = TargetEncoder()
        data[:, i:i+1] = enc.fit_transform(data[:, i:i+1], y)
        data_test[:, i:i+1]  = enc.transform(data_test[:, i:i+1])
    return data, data_test

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

# Naive Bayes classifier
def NB(X_train, X_val, y_train, y_val):
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB().fit(X_train, y_train)
    val_predictions = gnb.predict(X_val)

    # Validation accuracy
    accuracy = gnb.score(X_val, y_val)
    print('NB accuracy')
    print(accuracy)
    # Validation confusion matrix
    cm = confusion_matrix(y_val, val_predictions)
    print('NB CFmap')
    print(cm)

    #testing
    test_predictions = gnb.predict(x_test)
    return test_predictions
# KNN
def KNN(X_train, X_val, y_train, y_val):
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train)
    # model accuracy for X_test 
    accuracy = knn.score(X_val, y_val)
    print('KNN accuracy')
    print(accuracy)
    # creating a confusion matrix
    knn_predictions = knn.predict(X_val)
    cm = confusion_matrix(y_val, knn_predictions)
    print('KNN CFmap')
    print(cm)
    #testing
    test_predictions = knn.predict(x_test)
    return test_predictions
# SVM
def SVM(X_train, X_val, y_train, y_val, grid = False):
    from sklearn.svm import SVC
    
    # grid search
    if grid == True:
        svm = SVC()
        parameters = {'kernel':['rbf'],
                    'gamma': [ 1e-4, 1e-3, 1e-2], #, 1e-1, 1
                    'C': [2e7, 2e8, 2e9, 2e10, 2e11, 2e12, 2e13], #
                    }
        clf = GridSearchCV(svm, parameters)#  scoring=['recall_macro', 'precision_macro'], refit=False
        clf.fit(X_train, y_train)
        print(clf.best_params_)
        svm = clf.best_estimator_
    
    if grid == False:
        svm = SVC(kernel = 'rbf', C = 1)
        svm.fit(X_train, y_train)

    svm_predictions = svm.predict(X_val)
    accuracy = svm.score(X_val, y_val)
    print('SVM accuracy')
    print(accuracy)
    cm = confusion_matrix(y_val, svm_predictions)
    print('SVM CFmap')
    print(cm)
    print('RF f1 score')
    print(f1_score(y_val, svm_predictions, average = 'macro'))
    #testing
    test_predictions = svm.predict(x_test)
    return test_predictions
# Random Forest
def RF(X_train, X_val, y_train, y_val, grid = False):
    from sklearn.ensemble import RandomForestClassifier

    #grid search
    if grid == True:
        rf = RandomForestClassifier()
        parameters = {'n_estimators':range(100, 1000, 100)}
        clf = GridSearchCV(rf, parameters)
        clf.fit(X_train, y_train)
        rf = clf.best_estimator_

    if grid == False:
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
    
    #result
    rf_predictions = rf.predict(X_val)
    accuracy = rf.score(X_val, y_val)
    print('RF accuracy')
    print(accuracy)
    cm = confusion_matrix(y_val, rf_predictions)
    print('RF CFmap')
    print(cm)
    print('RF f1 score')
    print(f1_score(y_val, rf_predictions, average = 'macro'))
    #testing
    test_predictions = rf.predict(x_test)
    return test_predictions

def GradientBoosting(X_train, X_val, y_train, y_val, grid = False):
    from sklearn.ensemble import GradientBoostingClassifier

    if grid == True:
        gb = GradientBoostingClassifier()
        parameters = {'n_estimators': range(100, 500, 100),
                      'learning_rate': [0.01, 0.05, 0.1, 0.5, 1]
                      }
        clf = GridSearchCV(gb, parameters)
        clf.fit(X_train, y_train)
        print(clf.best_params_)
        gb = clf.best_estimator_

    else:
        gb = GradientBoostingClassifier(n_estimators = 400, learning_rate = 0.5, random_state = 1126)
        gb = Pipeline([ #('pca', PCA(n_components = 5)),
                        ('clf', gb)
                    ])
        gb.fit(X_train, y_train)

    # Validation
    accuracy = gb.score(X_val, y_val)
    print('GB accuracy')
    print(accuracy)
    # creating a confusion matrix
    gb_predictions = gb.predict(X_val)
    cm = confusion_matrix(y_val, gb_predictions)
    print('GB CFmap')
    print(cm)
    print('GB f1 score')
    print(f1_score(y_val, gb_predictions, average = 'macro'))
    # add validation-set into sub-training set
    X_trainval = np.vstack((np.array(X_train), np.array(X_val)))
    y_trainval = np.hstack((np.array(y_train), np.array(y_val)))

    # retraining w/ full training set
    if grid == True:
        gb = GradientBoostingClassifier(clf.best_params_)
    else:
        gb = GradientBoostingClassifier(n_estimators = 400, learning_rate = 0.5, random_state = 1126)

    gb.fit(X_trainval, y_trainval)
    # testing
    test_predictions = gb.predict(x_test)
    return test_predictions

# output prediction 
def make_pred(pred, test_id):
    with open('pred.csv', 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['Customer ID', 'Churn Category'])
        for i, p in enumerate(pred): 
            writer.writerow([test_id[i, 0], int(p)])

data_trainval, train_id, data_test, test_id = data_loader('trainval_data.csv','Test_data.csv')
y = np.ravel(encode_y(data_trainval))
# feats - numerical, feats_la - non-numerical
feats = [19, 22] + list(range(38,44))
feats_la =  [15, 17, 20, 21, 24, 25, 34, 35] + list(range(27,31))

# encode non-numerical data
data_trainval, data_test = encode_other(data_trainval, data_test, feats_la, y)

# feature extraction - numerical
# 19-Tenure_in_Months, 22-Avg_Monthly_Long_Distance_Charges,
# 38-Monthly_Charge, 39-Total_Charges...43 money-related data
x_trainval, x_test = feature_extractor(data_trainval, data_test, feats)

# do normalization
x_trainval, x_test = normalize(x_trainval, x_test)

# feature extraction - non-numerical
# 15-Satisfication_Score, 17-Referred_a_Friend, 20-Offer, 21-Phone_Service, 35-contract
x_trainval_la, x_test_la = feature_extractor(data_trainval, data_test, feats_la)
x_trainval = np.hstack((x_trainval, x_trainval_la))
print('number of features: {:4d}' 
        .format(len(x_trainval[0])))
x_test = np.hstack((x_test, x_test_la))

# Split data
X_train, X_val, y_train, y_val = train_test_split(x_trainval, y, random_state = 1126, train_size = 0.8)

# upsampling on training data only
from imblearn.over_sampling import SMOTE
smt = SMOTE(random_state = 1126)
X_train, y_train = smt.fit_resample(X_train, y_train)

#
from collections import Counter
print('number of samples:')
print(sorted(Counter(y_train).items()))

# predictions_NB = NB(X_train, X_val, y_train, y_val)
# predictions_KNN = KNN(X_train, X_val, y_train, y_val)
# predictions_RF = RF(X_train, X_val, y_train, y_val, False)
# predictions_SVM = SVM(X_train, X_val, y_train, y_val, True)
predictions_GB = GradientBoosting(X_train, X_val, y_train, y_val, False)

make_pred(predictions_GB, test_id)
