# f_class
'''
test = SelectKBest(score_func=f_classif, k=8)
fit = test.fit(x_trainval, y)
# summarize scores
set_printoptions(precision=3)
print(fit.scores_)
print(fit.get_support())
for i in range(len(fit.get_support())):
  if(fit.get_support()[i]==True):
    print(i," ",df.columns[i+1])
x_trainval = fit.transform(x_trainval)
x_test = fit.transform(x_test)
'''

#LogisticRegression
'''
  selector = SelectFromModel(estimator=LogisticRegression()).fit(x_trainval, y)

  x_trainval = selector.transform(x_trainval)
  x_test = selector.transform(x_test)
  print(x_trainval)
  print(x_test)
'''
#embedded
'''
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(x_trainval, y)
model = SelectFromModel(lsvc, prefit=True)
x_trainval = model.transform(x_trainval)
x_test = model.transform(x_test)
print(model.get_support())
print(x_trainval)
print(x_test)
'''
#K-neighbors
'''
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

knn = KNeighborsClassifier(n_neighbors=20)
sfs = SequentialFeatureSelector(knn, n_features_to_select=20)
sfs.fit(x_trainval, y)
SequentialFeatureSelector(estimator=KNeighborsClassifier(n_neighbors=20),
                          n_features_to_select=20)

print(sfs.get_support())
x_trainval = sfs.transform(x_trainval)
x_test = sfs.transform(x_test)
'''

#tree-based
'''
from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators=10)
clf = clf.fit(x_trainval, y)
#print(clf.feature_importances_)  
model = SelectFromModel(clf, prefit=True)

x_trainval = model.transform(x_trainval)
x_test = model.transform(x_test)
print(x_trainval.shape)  
print(x_test.shape)  
'''

#RFE
'''
from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

from sklearn.linear_model import LinearRegression

estimator = LinearRegression()
selector = RFE(estimator, n_features_to_select=8, step=1)
selector = selector.fit(x_trainval, y)
print(selector.support_)
for i in range(len(selector.support_)):
  if(selector.support_[i]==True):
    print(i," ",df.columns[i+1])

print(x_trainval.shape)
x_trainval = selector.transform(x_trainval)
print(x_trainval.shape)
x_test = selector.transform(x_test)
'''