import numpy
from pandas import read_csv
from sklearn.decomposition import PCA
# load data
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv('pima-indians-diabetes.csv', names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# feature extraction
pca = PCA(n_components=3)
fit = pca.fit(X)
# summarize components
print("Explained Variance: %s") % fit.explained_variance_ratio_
print(fit.components_)

features=pca.transform(X)
# summarize selected features
print('original data')
print(X[0:5,:])
print('transformed data')
print(features[0:5,:])