# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)
from pandas import read_csv
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv('pima-indians-diabetes.csv', names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# feature extraction
test = SelectKBest(score_func=chi2, k=4)
test.fit(X, Y)
# summarize scores
numpy.set_printoptions(precision=3)
print('features scores')
print(test.scores_)
features = test.transform(X)
# summarize selected features
print('original data')
print(X[0:5,:])
print('transformed data')
print(features[0:5,:])