from random import seed
from week6.csv_dataset import load_csv, str_column_to_float
from week6.csv_dataset import dataset_minmaxmean, normalize_inputdataset
from week6.csv_dataset import split_into_training_test_set, split_into_x_and_y
from sknn.mlp import Regressor, Layer
import numpy as np
import datetime

def create_network(niter, lr, vf):
    nn = Regressor(
        layers=[
            Layer("Sigmoid", units=5),
            Layer("Linear")],
        learning_rate=lr,
        n_iter=niter, verbose=vf)

    return nn


# Model Evaluation - RMSE
def rmse_score(Y, Y_pred):
    y_min_y_pred=Y - Y_pred
    rmse = np.sqrt(sum(y_min_y_pred ** 2) / float(len(Y)))
    return rmse

# Model Evaluation - R2 Score
def r2_score(Y, Y_pred):
    mean_y = np.mean(Y)
    ss_tot = sum((Y - mean_y) ** 2)
    ss_res = sum((Y - Y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


if __name__ == "__main__":
    seed(datetime.datetime.utcnow())
    filename = '/data/presentation/eml/python/src/week6/airfoil_self_noise.csv'
    dataset = load_csv(filename)
    for i in range(len(dataset[0]) ):
        str_column_to_float(dataset, i)
    minmax = dataset_minmaxmean(dataset)
    print('stats dataset', minmax)
    normalize_inputdataset(dataset, minmax)
    split=0.7
    trainingset, testset = split_into_training_test_set(dataset, split)
    train_x, train_y = split_into_x_and_y(trainingset)
    niter=100
    lr=0.01
    vf=True
    model=create_network(niter, lr, vf)
    model.fit(train_x, train_y)
    predicted_train_y=model.predict(train_x)
    rmse = rmse_score(train_y, predicted_train_y)
    r2 = r2_score(train_y, predicted_train_y)
    print("RMSE trainingset", rmse, "R2 Score trainingset", r2)
    test_x, test_y = split_into_x_and_y(testset)
    predicted_test_y = model.predict(test_x)
    rmse = rmse_score(test_y, predicted_test_y)
    r2 = r2_score(test_y, predicted_test_y)
    print("RMSE testset", rmse,"R2 Score testset", r2)
