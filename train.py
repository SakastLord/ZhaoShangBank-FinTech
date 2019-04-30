from model import Network
from prepare_data_op import *
import tensorflow as tf



def compute_acc_rec(predicted_list,truth_list):
    # 样本中： 1 是
    TP,FN,FP,TN = 0,0,0,0
    ALL_COUNT = len(predicted_list)
    for i in range(ALL_COUNT):
        prediction,truth = predicted_list[i],truth_list[i]
        if prediction == truth:
            if prediction == 0:
                # 负样本 预测为 负类
                TN = TN + 1
            if prediction == 1:
                # 正样本 预测为 正类
                TP = TP + 1
        else:
            if prediction == 1:
                # 负样本 预测为 正类
                FP = FP + 1
            else:
                # 正样本 预测为 负类
                FN = FN + 1
    acc     = ((TP + TN)*1.0)/ALL_COUNT
    rec     = (TP*1.0)/sum(truth_list)
    f_score = (acc*rec*2)/(acc+rec)
    return acc,rec,f_score,TP


def run():
    sess = tf.Session()
    raw_data = csv2pandas()
    train_labels,target_labels = get_train_test_columns()
    train_data,test_data = seperate_raw_and_train(raw_data,proportion=0.7)
    net = Network(sess=sess,n_input=len(train_labels),learning_rate=0.0001)

    test_x,test_y = sample_test_data(test_data=test_data,train_label=train_labels,target_label=target_labels)


    for epoch in range(100000):
        train_data,train_label = sample_train_data(train_data,batch_size=100,proportion=0.6,train_label=train_labels,test_label=target_labels)
        net.train(data_x=train_data,data_y=train_label,pro=1.0)
        if epoch %10==0:
            predicted_y_dirty = net.inference(data_x=test_x,pro=1.0)
            predicted_y = []
            for item in predicted_y_dirty:
                max_index = item.index(max(item))
                predicted_y.append(max_index)

