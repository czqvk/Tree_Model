import xgboost as xgb
import pandas as pd
from sklearn.externals import joblib

trdata_loc = r'D:\python\jupyter notebook\data\train.csv'
tr_data = pd.read_csv(trdata_loc)
tr_data['Sex'] = tr_data['Sex'].map({'male':0,'female':1})

feat_ls = ['Pclass','Sex','Age','Parch','Fare']
da_x = tr_data[feat_ls]
da_y = tr_data['Survived']

tr_x,te_x = da_x.iloc[:600],da_x.iloc[600:]
tr_y,te_y = da_y.iloc[:600],da_y.iloc[600:]


## 通用参数
xgb_default_params = {
'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 4, 'min_child_weight': 10, 'subsample': 0.8,
'colsample_bynode': 0.8, 'colsample_bylevel' : 1,'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
'base_score': 0.5, 'gamma': 0, 'max_delta_step': 0, 'scale_pos_weight': 1,
'objective': 'binary:logistic', 'booster': None
}

clf = xgb.XGBClassifier(**xgb_default_params)
# clf.set_params(**param)

## 设置早停的话，才能看见最优迭代次数和最优分数和迭代效果,
bst = clf.fit(tr_x,tr_y, early_stopping_rounds=200, eval_set=[[tr_x,tr_y],[te_x,te_y]], eval_metric = 'auc')

joblib.dump(clf,'xgb.pkl')
at = joblib.load('xgb.pkl')
tt = pd.DataFrame(tr_x.iloc[1]).T
print(at.predict_proba(tt))

# print(bst.best_score)
# print(bst.best_iteration)
# print(bst.evals_result_)

# print(bst.get_params())
# a = bst.predict(te_x)


xgb_params_details = {
    "n_estimators" : "迭代次数，默认100",
    "learning_rate" : "学习率，控制学习速度，建议设置为0.01-0.2，默认值为0.3",
    "max_depth" : "树模型深度，可以控制树模型复杂度，建议为3-8范围，默认值为6",
    "min_child_weight" : "决定最小叶子节点样本权重和，与lgb的不大一样，大一点则避免过拟合，默认值为1",
    "subsample": "每棵树，随机采样的，可以用来加速训练和处理过拟合，建议设置0.5-1，默认值为1",
    "colsample_bytree": "每一颗树随机选择特征的比例，可以用来加速训练和处理过拟合，建议0.5-1，默认值为1",
    "colsample_bylevel": "每一级的随机选择特征的比例，可以用来加速训练和处理过拟合，建议0.5-1，默认值为1",
    "colsample_bynode": "每一分裂节点随机选择特征的比例，可以用来加速训练和处理过拟合，建议0.5-1，默认值为1",
    "reg_alpha" : "L1正则，建议0.01-1",
    "reg_lambda" : "L2正则，建议0.01-1",
    "base_score" : "对于所有样本预测为正样本的全局偏置，如果迭代次数够多，改变这个参数对结果不会有影响。\
                    设定为#(正样本)/#(所有样本),对结果没有多少影响,但是可以减少迭代的次数（未验证），默认值为0.5",
    "gamma" : "在节点分裂时，只有分裂后损失函数的值下降了，才会分裂这个节点。Gamma指定了节点分裂所需的最小损失函数下降值，默认为0，可以适当调整",
    "max_delta_step": "限制每棵树权重改变的最大步长，如果被赋予了某个正值，那么算法更加保守，默认为0，一般不调整",
    "scale_pos_weight": "在各类别样本十分不平衡时，把这个参数设定为一个正值，可以使算法更快收敛",
    "objective" :"损失函数，一般不去调整",
    "booster" : "决定使用哪种booster，可选gbtree、dart、gblinear，默认gbtree，一般不做修改",
    "missing" : "被模型视为空值的项，默认值为np.nan",
    "seed" : "种子数，便于复现对比不同参数的效果"
}