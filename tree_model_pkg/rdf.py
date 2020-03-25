from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import StratifiedKFold
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
rdf_default_params = {
    'n_estimators': 80, 'max_depth': 5, 'max_features': 0.4, 'min_samples_leaf': 10, 'min_samples_split': 12,
    'max_leaf_nodes': None, 'min_impurity_split': None,  'min_weight_fraction_leaf': 0.0,'criterion': 'gini',
    'class_weight': None, 'bootstrap': True, 'oob_score': True, 'random_state': None, 'verbose': 0, 'warm_start': False}

clf = RandomForestClassifier(**rdf_default_params)
# clf.set_params(**param)
# clf = RandomForestClassifier(**param)

bst = clf.fit(tr_x,tr_y)
# print(bst.feature_importances_)
print(bst.get_params())

## 保存模型
joblib.dump(clf,'rdf.pkl')
at = joblib.load('rdf.pkl')
tt = pd.DataFrame(tr_x.iloc[1]).T
print(at.predict_proba(tt))



# df_y =pd.DataFrame(tr_y)
# sf_kf = StratifiedKFold(n_splits=5)
# for train_idx,test_idx in sf_kf.split(tr_x,tr_y):
#     tr_x1 = tr_x.iloc[train_idx]
#     tr_y1 =  tr_y.iloc[train_idx]
#     te_x = tr_x.iloc[test_idx]
#     te_y = tr_y.iloc[test_idx]
    # te_x, te_y = tr_x.iloc[test_idx], tr_y.iloc[test_idx]




rdf_params_detail = {
    "n_estimators" : "控制弱学习器的数量，默认值是10",
    "max_depth" : "决策树最大深度，默认值是None，样本和特征多的话可以取值大一点",
    "max_features": "寻找最佳分割时考虑的特征数目，可以输入[int, float, auto/sqrt, log2, None]，默认是auto",
    "min_samples_leaf" : "叶子节点所需最少的样本数，可填int或float，float则是一个百分比，表示节点最少需总样本数*百分比个样本，默认值为1",
    "min_sample_split": "内部节点再划分所需最小样本数，默认是2，样本量很大时可以增大该值，也可float传入，效果同上",
    "max_leaf_nodes" : "最大叶子节点数，做限制可以防止过拟合，默认是None",
    "min_impurity_split" : "如果某节点的不纯度(基于基尼系数，均方差)小于这个阈值，则该节点不再生成子节点，默认是None",
    "min_impurity_decrease" : "如果节点的分裂导致的不纯度的下降程度小于这个节点的值，则该节点不再生成子节点，默认是0",
    "min_weight_fraction_leaf" : "一个叶子节点所需要的权重总和的最小加权分数。当sample_weight没有提供时，样本具有相同的权重，默认是0",
    "criterion" : "判断节点是否继续分裂采用的计算方法，可选[gini, entropy]，默认gini",
    "class_weight" : "类别的权值，[None, balanced, 自定义字典]，balance会根据样本比例自动设置权值",
    "bootstrap" : "是否使用boostrap有放回抽样，默认值为True",
    "oob_score" : "是否计算袋外得分，默认None",
    "random_state" : "随机种子,控制产生随机数的种子，便于比对不同参数的结果，默认None",
    "warm_ start" : "使用它我们就可以用一个建好的模型来训练额外的决定树，能节省大量的时间，对于高阶应用我们应该多多探索这个选项，默认False",
    "verbose" : "决定建模完成后对输出的打印方式，0不输出，1打印特定区域树输出结果，>1打印所有结果，默认0",
}

# a = bst.predict(te_x)
