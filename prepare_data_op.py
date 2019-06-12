from params import *
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from imblearn.over_sampling import SMOTE


def csv2pandas():
    raw_data_pd = pd.read_csv(DATA_PATH).dropna(axis=0)
    return raw_data_pd


def read_train_data():
    raw_data_pd = pd.read_csv(DATA_PATH).fillna(method='ffill')
    return raw_data_pd


def read_test_data():
    raw_data_pd = pd.read_csv(EVALUATE_PATH)
    raw_data_pd = raw_data_pd.fillna(method='ffill')
    return raw_data_pd


def one_hot_cut(raw_data, whe_train_data):
    cut_dict = cut_dict_old
    if raw_data.columns[0] == 'stockcode':
        raw_data = raw_data.drop(raw_data.columns[0], axis=1)
    if whe_train_data is False:
        cut_df = pd.DataFrame()
        for field in cut_dict.keys():
            cut_series = pd.cut(raw_data[field], cut_dict[field], right=True)
            onehot_df = pd.get_dummies(cut_series, prefix=field)
            cut_df = pd.concat([cut_df, onehot_df], axis=1)
        new_df = pd.concat([raw_data, cut_df], axis=1)
        if ONLY_BIN_FEATURE:
            new_df = cut_df
        else:
            new_df = pd.concat([raw_data, cut_df], axis=1)
    else:
        target_column = raw_data[raw_data.columns[-1]]
        raw_data = raw_data.drop(raw_data.columns[-1], axis=1)
        train_columns = raw_data.columns[0:-1]
        target_columns = raw_data.columns[-1]
        cut_df = pd.DataFrame()
        for field in cut_dict.keys():
            cut_series = pd.cut(raw_data[field], cut_dict[field], right=True)
            onehot_df = pd.get_dummies(cut_series, prefix=field)
            cut_df = pd.concat([cut_df, onehot_df], axis=1)
        if ONLY_BIN_FEATURE:
            new_df = pd.concat([cut_df, target_column], axis=1)
        else:
            new_df = pd.concat([raw_data, cut_df], axis=1)
            new_df = pd.concat([new_df, target_column], axis=1)
            # print(new_df.head())
    return new_df


def check_nan_count():
    raw_data = pd.read_csv(DATA_PATH)
    bad_sample = raw_data[raw_data['fake'] == 1]
    len_bad = len(bad_sample)  # 违约样本数量
    # 查看 正样本数量 即 fake == 1
    good_sample = raw_data[raw_data['fake'] == 0]
    len_good = len(good_sample)  # 不违约样本数量
    print('不删除 Nan—— 整体长度：%s || 违约样本数量：%s  || 不违约样本数量：%s' % (len(raw_data), len_bad, len_good))
    raw_data = pd.read_csv(DATA_PATH).dropna(axis=0)
    bad_sample = raw_data[raw_data['fake'] == 1]
    len_bad = len(bad_sample)  # 违约样本数量
    # 查看 正样本数量 即 fake == 1
    good_sample = raw_data[raw_data['fake'] == 0]
    len_good = len(good_sample)  # 不违约样本数量
    print(' 删除 Nan—— 整体长度：%s || 违约样本数量：%s  || 不违约样本数量：%s' % (len(raw_data), len_bad, len_good))


def arguement_data(data, train_labels, target_labels):
    x = data[train_labels]
    y = data[target_labels]
    bad_data = data[data[target_labels] == 1]
    good_data = data[data[target_labels] == 0]
    print('SMOTE之前 bad：%s  good:%s' % (len(bad_data), len(good_data)))
    smo = SMOTE(random_state=42)
    x_arguemented, y_arguemented = smo.fit_sample(x, y)
    print(target_labels)

    x_arguemented = pd.DataFrame(x_arguemented, columns=train_labels)  # 将数据转换为数据框并命名列名
    y_arguemented = pd.DataFrame(y_arguemented, columns=[target_labels])  # 将数据转换为数据框并命名列名
    result = pd.concat([x_arguemented, y_arguemented], axis=1)  # 按列合并数据框
    bad_arguemented = result[result[target_labels] == 1]
    good_arguemented = result[result[target_labels] == 0]

    print('SMOTE之后 bad：%s  good:%s' % (len(bad_arguemented), len(good_arguemented)))
    result = result.sample(frac=1)
    return result


def check_distribution(data, column_min, column_max):
    result, _ = np.histogram(data, bins=20, range=(column_min, column_max))
    return result


def compute_columns_means():
    result_mean, result_max, result_min = [], [], []
    train_columns, target_column = get_train_test_columns()
    raw_data = csv2pandas()
    for column in train_columns:
        result_min.append(raw_data[column].min())
        result_max.append(raw_data[column].max())
        result_mean.append(raw_data[column].mean())
    result_distance = [max_item - min_item for max_item, min_item in zip(result_max, result_min)]
    return result_mean, result_distance, result_min, result_max


def length_str(num, length=2):
    result = str(round(num, 2)) + (length - (len(str(round(num, 2))) - 2)) * '0'
    return result


def check_data(raw_data):
    """
        正样本 7544
        负样本 109
        总样本 7653
    """
    # 查看 负样本数量 即 fake == 1
    bad_sample = raw_data[raw_data['fake'] == 1]
    len_bad = len(bad_sample)
    # 查看 正样本数量 即 fake == 1
    good_sample = raw_data[raw_data['fake'] == 0]
    len_good = len(good_sample)
    print("bad_sample length:%s  good_sample length:%s" % (len(bad_sample), len(good_sample)))
    train_labels, _ = get_train_test_columns()
    list_mean, list_distance, list_min, list_max = compute_columns_means()

    for index, item in enumerate(train_labels):
        column_min = list_min[index]
        column_max = list_max[index]
        np_bad = np.array(bad_sample[item]).reshape([1, -1])
        bad_result = check_distribution(np_bad, column_min=column_min, column_max=column_max)
        np_good = np.array(good_sample[item]).reshape([1, -1])
        good_result = check_distribution(np_good, column_min=column_min, column_max=column_max)
        bad_result = [length_str(item * 1.0 / len_bad) for item in bad_result]
        good_result = [length_str(item * 1.0 / len_good) for item in good_result]

        print('------------------------------%s------------------------------' % item)
        print('good', good_result)
        print('bad ', bad_result)

        bad_min, bad_max = bad_sample[item].min(), bad_sample[item].max()
        good_min, good_max = good_sample[item].min(), good_sample[item].max()
        print(' bad_min : %8s ||  bad_max : %8s' % (bad_min, bad_max))
        print('good_min : %8s || good_max : %8s' % (good_min, good_max))

        # 进行异常点检测
        Q1 = np.percentile(good_sample[item], 35)
        Q3 = np.percentile(good_sample[item], 65)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        bad_outlier = bad_sample[(bad_sample[item] < Q1 - outlier_step) | (bad_sample[item] > Q3 + outlier_step)]
        good_outlier = good_sample[(good_sample[item] < Q1 - outlier_step) | (good_sample[item] > Q3 + outlier_step)]
        print('bad离群点 ：%5s  || good离群点 : %5s ' % (len(bad_outlier), len(good_outlier)))
        temp = input()

    return bad_sample, good_sample, raw_data


def get_train_test_columns(whe_cut):
    train_data = read_train_data()
    if whe_cut:
        train_data = one_hot_cut(train_data, whe_train_data=True)
    all_columns = train_data.columns
    if all_columns[0] == 'stockcode':
        train_columns = all_columns[1:-1]
    else:
        train_columns = all_columns[0:-1]
    target_columns = all_columns[-1]
    return train_columns, target_columns


def hybrid_data(raw_data, proportion):
    """
    :param raw_data:
    :param proportion: 好样本 ：坏样本
    :return:
    """
    bad, good, data = check_data(raw_data)
    len_bad, len_good, len_data = len(bad), len(good), len(data)
    count_bad = len_bad
    if int(count_bad * proportion) < len_good:
        count_good = int(count_bad * proportion)
    elif int(count_bad * proportion) >= len_good:
        count_good = len_good
    print('current proportion:%s ||  count_good:%s || count_bad:%s' % (proportion, count_good, count_bad))
    return count_bad, count_good


def seperate_test_and_train(data, proportion):
    """

    :param data:
    :param proportion:  train : all  (between 0 and 1)
    :return:
    """
    data_all = data.sample(frac=1)
    len_all = len(data)
    count_train = int(len_all * proportion)
    train_data = data_all[:count_train]
    test_data = data_all[count_train:]
    return train_data, test_data


def sample_train_data(data, batch_size, proportion, train_label, test_label):
    """
    :param data:
    :param batch_size:
    :param proportion: good/(good+bad)
    :return:
    """
    # print('用来抽取训练数据的训练集合总长度 :',len(data))
    data = data.sample(frac=1)
    if batch_size is None:
        data_x = np.array(data[train_label])
        data_y = np.array(data[test_label])
        return data_x, data_y
    else:
        source_bad_data = data[data['fake'] == 1]
        source_good_data = data[data['fake'] == 0]
        count_good = int(batch_size * proportion)  # 每个batch中 正样本的个数
        count_bad = batch_size - count_good
        #     if batch_size-count_good>0 and count_good>=len(source_bad_data):
        #         count_bad  = batch_size-count_good
        #     else:
        #         print('需要的正样本数量为：%s  实际现有正样本数量为:%s'%(count_good,len(source_bad_data)))
        #         print("batch_size: %s || 需要的正样本数量:%s "%(batch_size,count_good))
        #         return

        if count_bad < len(source_bad_data):
            result_bad = source_bad_data.sample(n=count_bad)
            result_good = source_good_data.sample(n=count_good)
            # print('本次抽取的 违约样本数量为:%s  ||  正常样本数量为:%s '%(count_bad,count_good))

            result = pd.concat([result_good, result_bad], axis=0, ignore_index=True).sample(frac=1.0)
            data_x = np.array(result[train_label])
            data_y = np.array(result[test_label])
            return data_x, data_y
        else:
            result_bad = source_bad_data.sample(frac=1)
            result_good = source_good_data.sample(n=count_good)

            result = pd.concat([result_good, result_bad], axis=0, ignore_index=True).sample(frac=1.0)
            data_x = np.array(result[train_label])
            data_y = np.array(result[test_label])
            return data_x, data_y


def sample_test_data(test_data, train_label, target_label):
    test_x = test_data[train_label]
    test_y = test_data[target_label]
    return np.array(test_x), np.array(test_y)


if __name__ == '__main__':
    a, b = get_train_test_columns()
    print(a)
    print(b)
#     train_data = read_train_data()
#     print('cut train data')
#     cut_train_data = one_hot_cut(train_data,whe_train_data=True)
#     train_columns=(cut_train_data.columns)

#     test_data  = read_test_data()
#     print('cut test data')
#     cut_test_data = one_hot_cut(test_data,whe_train_data=False)
#     test_columns=(cut_test_data.columns)

#     print('----------------------------')
#     for i in range(len(test_columns)):
#         print(train_columns[i] , '-----', test_columns[i]    )
#     print(train_columns[-1])





