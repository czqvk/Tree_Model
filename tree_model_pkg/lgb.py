import lightgbm as lgb
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

## 配置参数
## silent=1 不输出中间过程
lgb_default_params = {
    'boosting_type': 'gbdt', 'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5, 'num_leaves': 31, 'min_child_samples': 20,
     'min_child_weight': 0.001, 'colsample_bytree': 0.8, 'min_split_gain': 0.0, 'objective': None, 'subsample': 0.8,
     'subsample_freq': 4, 'reg_alpha': 0.1, 'reg_lambda': 0.1, 'silent': True, 'random_state': None, 'class_weight': None
 }

clf = lgb.LGBMClassifier(**lgb_default_params)

## 设置早停的情况下才能看见迭代的具体效果（如果想看迭代效果，可以将早停设置到一个很大的数）
bst = clf.fit(tr_x,tr_y, early_stopping_rounds=200, eval_set=[(tr_x,tr_y),(te_x,te_y)], eval_metric = 'auc')
## 统计验证集的验证结果
print(bst.get_params())
# print(bst.predict_proba([['1','1','28','3','99.99']]))

# joblib.dump(clf,'lgbmd.pkl')
at = joblib.load('lgbmd.pkl')
print(at.predict_proba([['1','1','28','3','99.99']]))
# bst['clf'].save_model('lgbmd.txt')
# gbm = lgb.Booster(model_file='model.txt')


# print(bst.evals_result_)
# print(bst.best_iteration_)
# print(bst.best_score_)

lgb_params_detail ={
    "boosting_type" : "['gbdt', 'rf', 'dart', 'goss']这四个可选择，默认gbdt",
    "n_estimators" : "迭代次数，默认100",
    "learning_rate" : "学习率，控制学习速度，默认值为0.1",
    "max_depth" : "树模型深度，可以控制树模型复杂度，建议为3-8范围，默认值为-1(无调整策略)",
	"num_leaves" : "因为LightGBM使用的是leaf-wise的算法，因此在调节树的复杂程度时，使用的是num_leaves而不是max_depth。\
                   大致换算关系:num_leaves = 2^(max_depth), max_depth和num_leaves只需要设置一个",
    "max_bin" : "调小max_bin的值可以提高模型训练速度，调大它的值和调大num_leaves起到的效果类似",
    "min_child_samples": "一个叶子上数据的最小数量，可以用来处理过拟合，默认值20",
    "min_child_weight" : "决定最小叶子节点样本权重和，默认0.001",
    "colsample_bytree": "每次迭代中随机选择特征的比例，可以用来加速训练和处理过拟合，建议0.5-1，默认值为1",
    "min_split_gain" : "执行切分的最小增益，最小0.1，默认值为0",
    "objective" : "损失函数，一般不去调整，Default: 'regression' for LGBMRegressor, 'binary' or 'multiclass' for LGBMClassifier",
    "subsample" : "不进行重采样的情况下随机选择部分数据，可以用来加速训练和处理过拟合，subsample得和subsample_freq必须同时设置，建议设置0.5-1，默认值为1",
	"subsample_freq" : "bagging的次数,0表示禁用bagging，非零值表示k意味着每k轮迭代进行一次bagging，建议设置3-5范围，默认值0",
    "reg_alpha" : "L1正则，建议0.01-0.3",
    "reg_lambda" : "L2正则，建议0.01-0.3",
    "random_state" : "随机种子,控制产生随机数的种子，便于比对不同参数的结果",
    "is_unbalance" : "解决样本不平衡，设置为True时会把负样本的权重设为：正样本数/负样本数。这个参数只能用于二分类",
    "class_weight" : "用于多分类, 设置为'balanced'则自动调整样本权重，默认None",
}

