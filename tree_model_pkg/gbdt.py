from sklearn.ensemble import GradientBoostingClassifier as gbdt
import pandas as pd
from sklearn.externals import joblib


trdata_loc = r'D:\python\jupyter notebook\data\train.csv'
tr_data = pd.read_csv(trdata_loc)
tr_data['Sex'] = tr_data['Sex'].map({'male':0,'female':1})
tr_data['Age'] = tr_data['Age'].fillna(30)

feat_ls = ['Pclass','Sex','Age','Parch','Fare']
da_x = tr_data[feat_ls]
da_y = tr_data['Survived']

tr_x,te_x = da_x.iloc[:600],da_x.iloc[600:]
tr_y,te_y = da_y.iloc[:600],da_y.iloc[600:]

## 配置参数
## gbdt不准传入nan空值
gbdt_default_params = {'n_estimators': 100, 'learning_rate': 0.1, 'subsample': 1.0, 'loss': 'deviance', 'max_features': 0.4,
              'max_depth': 4, 'min_samples_split': 10, 'min_weight_fraction_leaf': 0.0,  'min_samples_leaf': 8,
              'max_leaf_nodes': None, 'min_impurity_split': None, 'criterion': 'friedman_mse','random_state': None,
              'init': None, 'presort': 'auto', 'verbose': 0,'warm_start': False}
clf = gbdt()
clf.set_params(**gbdt_default_params)
# clf = gbdt(**param)

bst = clf.fit(tr_x,tr_y)
a = bst.predict(te_x)

## 保存模型
joblib.dump(clf,'model_pkl/gbdt.pkl')
at = joblib.load('model_pkl/gbdt.pkl')
tt = pd.DataFrame(tr_x.iloc[1]).T
print(at.predict_proba(tt))

# print(bst.feature_importances_)
# print(bst.get_params())

gbdt_params_detail = {
    "n_estimators" : "控制弱学习器的数量，默认值是100",
    "learning_rate" : "学习率，默认0.1",
    "subsample" : "子采样，取值为（0,1]，与随机森林不同，随机森林是有放回抽样，这里是不放回抽样，又有一部分样本去做训练，建议在（0.5,0.8]",
    "loss" : "损失函数，对于分类模型有deviance和exponential,deviance是对数似然损失函数，exponential是指数损失函数",
    "max_features" : "划分时最大的特征数，可以有[\"log2\",\"sqrt\",0.4]",
    "max_depth" : "决策树最大深度，默认值是3，样本和特征多的话可以取值大一点",
    "min_sample_split" : "内部节点再划分所需最小样本数，默认是2，样本量很大时可以增大该值",
    "min_ weight_ fraction_leaf" : "叶子节点所需的样本数占总样本数的比值，默认是1",
    "min_samples_leaf" : "叶子节点所需最少的样本数",
    "max_leaf_nodes" : "最大叶子节点数，做限制可以防止过拟合，默认是None",
    "min_impurity_split" : "如果某节点的不纯度(基于基尼系数，均方差)小于这个阈值，则该节点不再生成子节点",
    "random_state" : "随机种子,控制产生随机数的种子，便于比对不同参数的结果",
    "presort" : "决定是否对数据进行预排序，可以使得树分裂地更快",
    "warm_ start" : "使用它我们就可以用一个建好的模型来训练额外的决定树，能节省大量的时间，对于高阶应用我们应该多多探索这个选项",
    "verbose" : "决定建模完成后对输出的打印方式，0不输出，1打印特定区域树输出结果，>1打印所有结果",
    "init" : "如果我们有一个模型，它的输出结果会用来作为GBM模型的起始估计，这个时候就可以用init"
}