from model import *
from prepare_data_op import *
from evaluate import *
import tensorflow as tf
from params import *
import xgboost as xgb
from utils import *

EPI_COUNT = 1000000

WHE_LIMIT = True
WHE_BIN = False
WHE_ZERO_ONE = True


def get_train_test_columns():
    train_data = read_train_data()

    if WHE_BIN:
        train_data = one_hot_cut(train_data, whe_train_data=True)

    all_columns = train_data.columns

    if all_columns[0] == 'stockcode':
        train_columns = all_columns[1:-1]

    else:
        train_columns = all_columns[0:-1]
    target_columns = all_columns[-1]
    return train_columns, target_columns


def network_inference(model):
    evaluate_data = read_test_data()

    if WHE_LIMIT:
        evaluate_data = limit_test_data(evaluate_data)

    # 对训练数据进行归一化
    if WHE_ZERO_ONE:
        evaluate_data = between_zero_one(evaluate_data)

    # 进行 分箱 和 one-hot编码
    if WHE_BIN:
        evaluate_data = one_hot_cut(evaluate_data, whe_train_data=False)

    train_labels, target_labels = get_train_test_columns()
    result = model.inference(evaluate_data[train_labels])
    print('result:', sum(result))
    predict_count = sum(result)
    result_pd = pd.DataFrame(
        {'stockcode': np.array(pd.read_csv(EVALUATE_PATH)['stockcode']).tolist(),
         'fake': result}
    )
    # result_pd.to_csv(OUTPUT_PATH+'_5_4'+'_net_result.csv',index=False)
    return result_pd, predict_count


def compute_acc_rec(predicted_list, truth_list):
    # 样本中： 1 是
    TP, FN, FP, TN = 0, 0, 0, 0
    ALL_COUNT = len(predicted_list)
    for i in range(ALL_COUNT):
        prediction, truth = predicted_list[i], truth_list[i]
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
    if (TP + FP) > 0:
        P = (TP * 1.0) / (TP + FP)  # accuarcy
    else:
        P = 0
    if (TP + FN) > 0:
        R = (TP * 1.0) / (TP + FN)  # recall
    else:
        R = 0
    if (P + R) > 0:
        f_score = (P * R * 2) / (P + R)
    else:
        f_score = 0
    return P, R, f_score


def run():
    sess = tf.Session()

    train_data_raw = read_train_data()

    # 对训练数据进行 limit
    if WHE_LIMIT:
        train_data_raw = limit_train_data(train_data_raw)

    # 对训练数据进行归一化
    if WHE_ZERO_ONE:
        train_data_raw = between_zero_one(train_data_raw)

    # 进行 分箱 和 one-hot编码
    if WHE_BIN:
        train_data_raw = one_hot_cut(train_data_raw, whe_train_data=True)

    train_labels, target_labels = get_train_test_columns()

    # 划分 训练集 和 测试集
    train_data, test_data = seperate_test_and_train(train_data_raw, proportion=0.8)

    print('train_labels', train_labels)
    print('target_labels', target_labels)

    # 使用smote进行数据增强
    # train_data_arguemented = arguement_data(train_data,train_labels,target_labels)

    net = Network(sess=sess, n_input=len(train_labels))

    sess.run(tf.global_variables_initializer())

    # 测试集的 x 和 y
    test_x, test_y = sample_test_data(test_data=test_data, train_label=train_labels, target_label=target_labels)

    f_test_record, f_train_record = 0.0, 0.0

    # for epoch in range(EPI_COUNT):
    for epoch in range(100000):

        # 每个epoch 都从 训练集train_data中 抽取出x和y 条数总共为batch_size，其中proportion为 正样本:总样本
        train_x, train_y = sample_train_data(train_data, batch_size=100, proportion=0.8, train_label=train_labels,
                                             test_label=target_labels)

        net.train(data_x=train_x, data_y=train_y)

        if epoch % 100 == 0:
            print("-------------------------------------------------------------------------------------------")
            print('是否分箱：%s ||  是否裁剪：%s || 是否归一化：%s' % (WHE_BIN, WHE_LIMIT, WHE_ZERO_ONE))

            # 计算测试集 p,r,f_socre
            predicted_y_dirty = net.inference(data_x=test_x)
            p_test, r_test, f_test = compute_acc_rec(predicted_list=predicted_y_dirty, truth_list=test_y)
            if WHE_PRINT:
                print("模型: %s || 测试集——No:%6d || 准确率：%2.3f || 召回率：%2.3f || F_Score：%2.3f || 正样本总数:%2d "
                      "|| 预测的正样本总数:%2d || 总数据：%4s" % (
                          str(MODEL), epoch, p_test, r_test, f_test, sum(test_y), sum(predicted_y_dirty), len(test_x)))

            # 计算训练集 p,r,f_socre
            predicted_y_dirty = net.inference(data_x=np.array(train_data[train_labels]))
            p_train, r_train, f_train = compute_acc_rec(predicted_list=predicted_y_dirty,
                                                        truth_list=train_data[target_labels].tolist())
            if WHE_PRINT:
                print("模型: %s || 训练集——No:%6d || 准确率：%2.3f || 召回率：%2.3f || F_Score：%2.3f || 正样本总数:%2d "
                      "|| 预测的正样本总数:%2d || 总数据：%4s" % (
                          str(MODEL), epoch, p_train, r_train, f_train, sum(train_data[target_labels].tolist()),
                          sum(predicted_y_dirty), len(train_data)))

            # update f_test_record,f_train_record
            #             if f_test>=f_test_record and f_train>f_train_record and f_train>1.5:
            if f_test >= f_test_record and epoch >= 2500 and f_test > 0.12:
                f_test_record, f_train_record = round(f_test, 3), round(f_train, 3)
                result, count = network_inference(net)
                print(f_train_record, f_test_record)
                # net.save_current_model(f_train_record,f_test_record)
                current_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
                score_record = 'Model:' + str(MODEL) + '_L_reg:' + str(L_reg) + '_Train:' + str(
                    f_train_record) + '_Test:' + str(f_test_record) + '_BETA_REG:' + str(BETA_REG) + '_Count:' + str(
                    count) + '_Keep_prob:' + str(KEEP_PROB)
                result.to_csv(OUTPUT_PATH + 'result_' + str(current_time) + score_record + '_result.csv', index=False)



