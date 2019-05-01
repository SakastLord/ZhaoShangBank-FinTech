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
    P = (TP*1.0)/(TP+FP)     # accuarcy
    R = (TP*1.0)/(TP+FN) # recall
    f_score = (P*R*2)/(P+R)
    return P,R,f_score


def run():
    sess = tf.Session()
    raw_data = csv2pandas()
    train_labels,target_labels = get_train_test_columns()
    train_data,test_data = seperate_test_and_train(raw_data,proportion=0.7)

    net = Network(sess=sess,n_input=len(train_labels),learning_rate=0.0001)
    test_x,test_y = sample_test_data(test_data=test_data,train_label=train_labels,target_label=target_labels)
    for epoch in range(100000):
        train_x,train_y = sample_train_data(train_data,batch_size=100,proportion=0.6,train_label=train_labels,test_label=target_labels)
        net.train(data_x=train_x,data_y=train_y,prob=1.0)
        if epoch %10==0:
            # compute test data p,r,f_socre
            print("-------------------------------------------------------------------------------------------")

            predicted_y_dirty = net.inference(data_x=test_x,prob=1.0)
            predicted_y = []
            for item in predicted_y_dirty:
                max_index = item.index(max(item))
                predicted_y.append(max_index)
            p_test,r_test,f_test = compute_acc_rec(predicted_list=predicted_y,truth_list=test_y)

            print("测试集——No:%6d || 准确率：%2.3f || 召回率：%2.3f || F_Score：%2.3f || 正样本总数:%2d "
                  "|| 预测的正样本总数:%2d " % (
                  epoch, p_test, r_test, f_test, sum(test_y), sum(predicted_y)))

            # compute train data p,r,f_socre
            predicted_y_dirty = net.inference(data_x=train_x, prob=1.0)
            predicted_y = []
            for item in predicted_y_dirty:
                max_index = item.index(max(item))
                predicted_y.append(max_index)
            p_train,r_train,f_train = compute_acc_rec(predicted_list=predicted_y, truth_list=train_y)

            print("训练集——No:%6d || 准确率：%2.3f || 召回率：%2.3f || F_Score：%2.3f || 正样本总数:%2d "
                  "|| 预测的正样本总数:%2d " % (
                  epoch, p_train, r_train, f_train, sum(train_y.tolist()), sum(predicted_train)))


