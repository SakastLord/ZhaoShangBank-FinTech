import pandas as pd
import numpy as np
from prepare_data_op import *

train_data = read_train_data()

target_columns = train_data['fake']

for item in clean_dict.keys():
    min_item, max_item = clean_dict[item][2], clean_dict[item][3]
    train_data.loc[(train_data[item] <= max_item) & (train_data[item] >= min_item), [item]] = 1
    train_data.loc[(train_data[item] > max_item) | (train_data[item] < min_item), [item]] = 0

train_data_result = train_data[clean_dict.keys()]

print(train_data_result)

print('------------------------')

train_data_result['count'] = train_data_result.apply(lambda x: x.sum(), axis=1)

train_data_result = pd.concat([train_data_result, target_columns], axis=1)

print(train_data_result[train_data_result['fake'] == 0])

print(train_data_result[train_data_result['count'] > 200])







