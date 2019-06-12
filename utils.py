import pandas as pd
import numpy as np
from params import *
from prepare_data_op import *


def up_sample(data, count, train_labels, target_labels, whe_smote=True):
    if whe_smote:

        bad_Sample = data[data['fake'] == 1]
        len_bad_before = len(bad_Sample)

        x = data[train_labels]
        y = data[target_labels]
        bad_data = data[data[target_labels] == 1]
        good_data = data[data[target_labels] == 0]
        print('SMOTE之前 bad：%s  good:%s' % (len(bad_data), len(good_data)))
        smo = SMOTE(random_state=42)
        x_arguemented, y_arguemented = smo.fit_sample(x, y)
        x_arguemented = pd.DataFrame(x_arguemented, columns=train_labels)  # 将数据转换为数据框并命名列名
        y_arguemented = pd.DataFrame(y_arguemented, columns=[target_labels])  # 将数据转换为数据框并命名列名
        result = pd.concat([x_arguemented, y_arguemented], axis=1)  # 按列合并数据框
        bad_arguemented = result[result[target_labels] == 1]
        good_arguemented = result[result[target_labels] == 0]

        bad_Sample = bad_arguemented.sample(n=len_bad_before)
        for i in range(count):
            data = data.append(bad_Sample)
        data = data.sample(frac=1)
        bad_Sample = data[data['fake'] == 1]
        len_bad_after = len(bad_Sample)
        print('过采样前 训练集有%s条正样本 ||  过采样后 训练集有%s条正样本 ' % (len_bad_before, len_bad_after))
        return data
    else:
        bad_Sample = data[data['fake'] == 1]
        good_Sample = data[data['fake'] == 0]
        len_bad_before = len(bad_Sample)
        for i in range(count):
            data = data.append(bad_Sample)
        data = data.sample(frac=1)
        bad_Sample = data[data['fake'] == 1]
        len_bad_after = len(bad_Sample)
        print('过采样前 训练集有%s条正样本 ||  过采样后 训练集有%s条正样本 ' % (len_bad_before, len_bad_after))
        return data


def limit_train_data(data):
    if str(data.columns[0]) == 'stockcode':
        data = data.drop(data.columns[0], axis=1)  # 丢弃 stock_id 列
    good_samples = data[data['fake'] == 0]
    bad_samples = data[data['fake'] == 1]
    for field in clean_dict.keys():
        good_min, good_max, bad_min, bad_max = clean_dict[field][0], clean_dict[field][1], clean_dict[field][2], \
                                               clean_dict[field][3]
        # limit the good_samples
        good_samples.loc[good_samples[field] > good_max, field] = good_max
        good_samples.loc[good_samples[field] < good_min, field] = good_min
        # limit the bad_samples
        bad_samples.loc[bad_samples[field] > bad_max, field] = bad_max
        bad_samples.loc[bad_samples[field] < bad_min, field] = bad_min
    limit_df = pd.concat([good_samples, bad_samples], axis=0)
    limit_df = limit_df.sample(frac=1)
    return limit_df


def limit_test_data(data):
    if str(data.columns[0]) == 'stockcode':
        data = data.drop(data.columns[0], axis=1)  # 丢弃 stock_id 列
    for field in clean_dict.keys():
        good_min, good_max, bad_min, bad_max = clean_dict[field][0], clean_dict[field][1], clean_dict[field][2], \
                                               clean_dict[field][3]
        temp_list = [good_min, good_max, bad_min, bad_max]
        max_item, min_item = max(temp_list), min(temp_list)
        data.loc[data[field] > max_item, field] = max_item
        data.loc[data[field] < min_item, field] = min_item
    data = data.sample(frac=1)
    return data


def between_zero_one(data):
    for field in min_max_mean_std_dict:
        mean_item, std_item = min_max_mean_std_dict[field][2], min_max_mean_std_dict[field][3]
        data[field] = (data[field] - mean_item) / std_item
    data['Neg_Dednp_times'] = data['Neg_Dednp_times'] / 3.0
    return data









