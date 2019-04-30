import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def csv2pandas():
    raw_data_pd = pd.read_csv('/clever/data/FT_Camp_5/Train.csv')
    return raw_data_pd



def check_distribution(data,title,xlabel,ylabel):
    # plt.hist(data)
    # plt.xlabel(xlabel=xlabel)
    # plt.ylabel(xlabel=ylabel)
    # plt.title(title)
    # plt.show
    # N = 20
    # max,min = data.max(),data.min()
    # distance = (max - min)*1.0/N
    # temp = [distance*x for x in range(0,N)]
    # result = [0]*N
    # for item in data:
    #     if
    result = np.histogram(data,bins=20)
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
    print("bad_sample length:%s  good_sample length:%s"%(len(bad_sample),len(good_sample)))
    for item in raw_data.columns:
        np_bad  = np.array(bad_sample[item]).reshape([1,-1])
        bad_result = check_distribution(np_bad,title='bad',xlabel=item,ylabel='count')
        np_good = np.array(good_sample[item]).reshape([1,-1])
        good_result = check_distribution(np_good,title='good',xlabel=item,ylabel='count')
        print("good : ",good_result)
        print("bad  : ",bad_result)
    temp = input()

    return bad_sample , good_sample , raw_data


def get_train_test_columns():
    raw_data_pd = pd.read_csv('/clever/data/FT_Camp_5/Train.csv')
    all_columns = raw_data_pd.colummns
    train_columns  = all_columns[1:-1]
    target_columns = all_columns[-1]
    return train_columns,target_columns




def hybrid_data(raw_data,proportion):
    """
    :param raw_data:
    :param proportion: 好样本 ：坏样本
    :return:
    """
    bad,good,data = check_data(raw_data)
    len_bad,len_good,len_data = len(bad),len(good),len(data)
    count_bad = len_bad
    if int(count_bad*proportion) < len_good:
        count_good = int(count_bad*proportion)
    elif int(count_bad*proportion) >= len_good:
        count_good = len_good
    print('current proportion:%s ||  count_good:%s || count_bad:%s'%(proportion,count_good,count_bad))
    return count_bad,count_good



def seperate_raw_and_train(data,proportion):
    """

    :param data:
    :param proportion:  train : all  (between 0 and 1)
    :return:
    """
    data_all = data.sample(frac = 1)
    len_all = len(data)
    bad_data  = data[raw_data['fake'] == 1]
    good_data = data[raw_data['fake'] == 0]
    count_train = int(len_all*proportion)
    count_test = int(len_all - count_train)
    train_data = data_all[:count_train]
    test_data  = data_all[count_train:]
    return train_data,test_data


def sample_train_data(data,batch_size,proportion,train_label,test_label):
    """
    :param data:
    :param batch_size:
    :param proportion: good/(good+bad)
    :return:
    """
    source_bad_data  = data[data['fake'] == 1]
    source_good_data = data[data['fake'] == 0]
    count_good = batch_size*proportion # 每个batch中 正样本的个数
    count_bad  = batch_size-count_good
    result_bad  = source_bad_data.sample(n = count_bad)
    result_good = source_good_data.sample(n = count_good)
    result = pd.concat([result_good, result_bad], axis=0, ignore_index=True).sample(frac=1.0).reset_index()
    return np.array(result[train_label]) , np.array(result[test_label])


def sample_test_data(test_data,train_label,target_label):
    test_x = test_data[train_label]
    test_y = test_data[target_label]
    return np.array(test_x) , np.array(test_y)






if __name__ == '__main__':
    # raw_data = csv2pandas()
    # check_data(raw_data)
    train_columns,test_columns = get_train_test_columns()
    print(train_columns,test_columns)



