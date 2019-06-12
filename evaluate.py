import pandas as pd
import numpy as np
import time
import xlab
from sklearn.externals import joblib
from params import *
from prepare_data_op import *


def read_evaluate_data():
    raw_data_pd = pd.read_csv(EVALUATE_PATH)
    raw_data_pd = raw_data_pd.fillna(0)
    return raw_data_pd


def load_model(model_one_path, model_two_path):
    model_one = joblib.load(model_one_path)
    model_two = joblib.load(model_two_path)
    return model_one, model_two


def normalize_evaluate_data():
    train_labels, _ = get_train_test_columns()
    # result_mean, result_distance, result_min , _ = compute_columns_means()
    eval_data = read_test_data()
    eval_data = one_hot_cut(eval_data, whe_train_data=False)
    eval_data = eval_data[train_labels]
    #     for index, column in enumerate(train_labels):
    #         eval_data[column] = (eval_data[column] - result_min[index]) / result_distance[index]
    return np.array(eval_data)


def compute_fusion(x, y):
    if x == 0 and y == 0:
        return 0
    else:
        return 1


def evaluate_inference(model_one_path, model_two_path):
    evaluate_data = normalize_evaluate_data()
    print(evaluate_data)
    model_one, model_two = load_model(model_one_path, model_two_path)
    result_one = model_one.predict(evaluate_data)
    result_two = model_two.predict(evaluate_data)
    result = [compute_fusion(x, y) for x, y in zip(result_one, result_two)]
    return result, result_one, result_two


def generate_evaluate_result():
    model_one_path = MODEL_SAVE_PATH + '2019-05-04 09:22:19_Train:0.362_Test:0.646_xgboost.m'
    model_two_path = MODEL_SAVE_PATH + '2019-05-04 09:22:19_Train:0.362_Test:0.646_tree.m'
    result, one, two = evaluate_inference(model_one_path, model_two_path)
    print("model fusion 抓出的正样本", sum(result))
    print("model one 抓出的正样本", sum(one))
    print("model two 抓出的正样本", sum(two))
    result_pd = pd.DataFrame({'stockcode': np.array(pd.read_csv(EVALUATE_PATH)['stockcode']).tolist(),
                              'fake': result})
    current_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    result_pd.to_csv(OUTPUT_PATH + str(current_time) + '_result.csv', index=False)


def submit(filename):
    # 提交结果 filename
    xlab.ftcamp.submit(OUTPUT_PATH + filename)


def check_history():
    xlab.ftcamp.get_submit_hist()


if __name__ == '__main__':
    # generate_evaluate_result()
    # filename = '2019-05-03_21:11:18_result.csv'
    # submit(filename)
    check_history()








