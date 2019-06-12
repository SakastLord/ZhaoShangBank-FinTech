from model import *
from prepare_data_op import *
from evaluate import *
import tensorflow as tf
from params import *
import xgboost as xgb
from utils import *
from sklearn.model_selection import GridSearchCV


def compute_acc_rec(predicted_list, truth_list):
    TP, FN, FP, TN = 0, 0, 0, 0
    ALL_COUNT = len(predicted_list)
    for i in range(ALL_COUNT):
        prediction, truth = predicted_list[i], truth_list[i]
        if prediction == truth:
            if prediction == 0:
                TN = TN + 1
            if prediction == 1:
                TP = TP + 1
        else:
            if prediction == 1:
                FP = FP + 1
            else:
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


if __name__ == '__main__':
    train_data_raw = read_train_data()

    train_data_raw = limit_train_data(train_data_raw)

    # 归一化
    # train_data_raw = between_zero_one(train_data_raw)

    train_data_raw = one_hot_cut(train_data_raw, whe_train_data=True)

    train_labels, target_labels = get_train_test_columns()

    # seperate train and test
    train_data, test_data = seperate_test_and_train(train_data_raw, proportion=0.7)

    # seperate test-set into Label and target
    test_x, test_y = np.array(test_data[train_labels]), np.array(test_data[target_labels])

    # over-sample
    # train_data_up_sample = up_sample(train_data,40)

    # seperate train-set into Label and target
    train_x, train_y = np.array(train_data_raw[train_labels]), np.array(train_data_raw[target_labels])

    other_params = {'eta': 0.3, 'n_estimators': 500, 'gamma': 0, 'max_depth': 6, 'min_child_weight': 1,
                    'colsample_bytree': 1, 'colsample_bylevel': 1, 'subsample': 1, 'reg_lambda': 1, 'reg_alpha': 0,
                    'seed': 33}

    cv_params = {'n_estimators': np.linspace(100, 1000, 10, dtype=int),
                 'max_depth': np.linspace(3, 10, 7, dtype=int),
                 'min_child_weight': np.linspace(1, 10, 10, dtype=int),
                 #                  'gamma':np.linspace(0, 1, 10),
                 #                  'subsample': np.linspace(0, 1, 11),
                 #                  'colsample_bytree': np.linspace(0, 1, 11)[1:],
                 #                  'reg_lambda': np.linspace(0, 100, 11),
                 #                  'reg_alpha': np.linspace(0, 10, 11),
                 #                  'eta': np.logspace(-2, 0, 10),
                 }

    model = xgb.XGBClassifier(**other_params)  # 注意这里的两个 * 号！
    gs = GridSearchCV(model, cv_params, verbose=2, refit=True, cv=4, n_jobs=-1, scoring='f1', )
    gs.fit(train_x, train_y)  # X为训练数据的特征值，y为训练数据的label
    # 性能测评
    print("参数的最佳取值：:", gs.best_params_)
    print("最佳模型得分:", gs.best_score_)

#     model_name = 'xgboost'

#     model = XGBClassifier(booster = 'gbtree',
#                         #scale_pos_weight=10,
#                         n_estimatores = 50,
#                         max_depth = 5,
#                         learning_rate = 0.1,
#                         objective = 'binary:logitraw',
#                         #gamma=0.2,
#                         reg_lambda=1,
#                         silent=0,
#                         colsample_bytree = 0.6,
#                         class_weight = 'balanced'
#                         )

#     model.fit(train_x_up_sample,train_y_sample )

#     # 计算测试集 p,r,f_socre
#     predicted_y_test = model.predict(test_x)
#     p_test,r_test,f_test = compute_acc_rec(predicted_list=predicted_y_test,truth_list=test_y)
#     print("模型: %s || Test  || Acc：%2.3f || Recall：%2.3f || F_Score：%2.3f || 实际正样本总数:%2d || 预测的正样本总数:%2d || 总数据：%4s" % (
#             model_name, p_test,  r_test,  f_test, sum(test_y),  sum(predicted_y_test) , len(test_x)))

#     # 计算训练集 p,r,f_socre
#     predicted_y_train = model.predict(train_x)
#     p_train,r_train,f_train = compute_acc_rec(predicted_list=predicted_y_train, truth_list=train_data[target_labels].tolist())
#     print("模型: %s || Train || Acc：%2.3f || Recall：%2.3f || F_Score：%2.3f || 实际正样本总数:%2d || 预测的正样本总数:%2d || 总数据：%4s" % (
#             model_name, p_train, r_train, f_train, sum(train_y),sum(predicted_y_train) ,len(train_x)))

#             # update f_test_record,f_train_record

#     if f_test>0.1:

#         evaluate_data = read_test_data()

#         evaluate_data = limit_test_data(evaluate_data)

#         evaluate_data = between_zero_one(evaluate_data)

#         evaluate_data = one_hot_cut(evaluate_data,whe_train_data=False)

#         evaluate_data = evaluate_data[train_labels]

#         evaluate_data = np.array(evaluate_data)

#         result = model.predict(evaluate_data)

#         print('针对3000多条测试集，预测出的正样本共有%s条'%sum(result))

#         count = sum(result)

#         result = pd.DataFrame({'stockcode':np.array(pd.read_csv(EVALUATE_PATH)['stockcode']).tolist(),'fake':result})

#         f_test_record,f_train_record = round(f_test,3), round(f_train,3)

#         print(f_train_record,f_test_record)

#         current_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

#         score_record_one = 'Model:'+str(MODEL)+'_L_reg:'+str(L_reg)+'_Train:'+str(f_train_record)+'_Test:'+str(f_test_record)

#         score_record_two = '_BETA_REG:'+str(BETA_REG)+'_Count:'+str(count)+'_Keep_prob:'+str(KEEP_PROB)

#         filepath = OUTPUT_PATH+'result_'+str(current_time)+score_record_one+score_record_two+'_result.csv'

#         result.to_csv(filepath,index=False)
