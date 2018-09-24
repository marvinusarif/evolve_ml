from pandas import read_csv
import numpy

def step1():
    dataset = read_csv('pima-indians-diabetes.csv', header=None)
    print(dataset.describe())


def step2():
    dataset = read_csv('pima-indians-diabetes.csv', header=None)
    # print the first 20 rows of data
    print(dataset.head(20))


def step3_normalizing():
    dataset = read_csv('pima-indians-diabetes.csv', header=None)
    # print the first 20 rows of data
    #normalized_df = (dataset - dataset.mean()) / dataset.std()

    normalized_df = (dataset - dataset.min()) / (dataset.max() - dataset.min())
    print('sample dataset')
    print(normalized_df.head(20))
    print('standard deviasi original data')
    print(dataset.std())
    print('standard deviasi')
    print(normalized_df.std())
    print('standard deviasi > 0.12')
    print(normalized_df.std()>0.12)


step3_normalizing()
