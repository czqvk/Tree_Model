import xgboost
import lightgbm
import xgboost as xgb
import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV,KFold,StratifiedKFold,train_test_split
from sklearn.metrics import roc_curve,auc

class TreeModel():
    def __init__(self):
        '''
        分类预测，特征重要性、K折验证、建立各模型默认参数列表、lgb和xgb的clf可以查看迭代效果，各模型参数参考, 输出模型参数大全及说明
        保存模型读取模型
        '''
        ## 模型通用参数
        self.model_defalt_param = model_defalt_param
        ## 模型参数解释
        self.model_param_detail = model_param_detail

    def __model_build_param(self, param, model_name):
        '''
        :return: 传入参数，可使用默认参数也可自传参数
        '''
        defalt_param_d = model_defalt_param.get(model_name)
        param_input = defalt_param_d.copy()
        if not param:
            ## 没有参数传入则采用默认参数
            print("使用自设通用参数")
        else:
            ## 传入参数则在默认参数基础上添加或修改
            if isinstance(param,dict):
                for param_name,param_value in param.items():
                    param_input[param_name] = param_value
            else:
                print("传入模型参数应为字典格式传入，此次先采用默认参数")
        self.model.set_params(**param_input)


    def __model_pick(self, model_name):
        '''
        :param model_name: 确认模型，初始化模型
        :return:
        '''
        model_name_ls = ['lgb','xgb','gbdt','rdf']
        tree_model_ls = [lgb.LGBMClassifier(),xgb.XGBClassifier(),GradientBoostingClassifier(),RandomForestClassifier()]
        if model_name in model_name_ls:
            model_idx = model_name_ls.index(model_name)
            clf_model = tree_model_ls[model_idx]
            self.lgb,self.xgb,self.gbdt,self.rdf = [False]*4
            if model_name == 'lgb':
                self.lgb = True
            elif model_name == 'xgb':
                self.xgb = True
            elif model_name == 'gbdt':
                self.gbdt = True
            elif model_name == 'rdf':
                self.rdf = True
            return clf_model
        else:
            raise Exception('传入的模型名称有误，请在[\'lgb\',\'xgb\',\'gbdt\',\'rdf\']中选定')


    def __data_null_check(self,uncheck_data):
        '''
        :return: 所有训练的特征数据需要以DataFrame传入，如果模型是gbdt和rdf的话，需要检查数据是否有空值
        '''
        if not isinstance(uncheck_data,pd.DataFrame):
            raise Exception("传入的训练数据不是pandas.DataFrame格式")
        else:
            if self.gbdt or self.rdf:
                na_num = sum(uncheck_data.isnull().sum().values)
                if na_num > 0:
                    raise Exception("传入的训练数据有空值，GBDT和随机森林要求非空传入")
            else:
                pass


    def __fea_importance(self, clf, feat_name):
        '''
        :param clf: fit后的分类器
        :param feat_name: 特征名
        :return: 分类器fit后输出特征重要性
        '''
        if self.lgb:
            ## lightgbm的特征重要性接口与其他的不同
            fea_imp_int = clf.booster_.feature_importance()
            feat_imp = [round(imp/sum(fea_imp_int),4) for imp in fea_imp_int]
        else:
            feat_imp = [round(imp,4) for imp in clf.feature_importances_]
        feat_imp_dict = {feat_name[i]:feat_imp[i] for i in range(len(feat_imp))}
        return feat_imp_dict


    def fit(self, train_x, train_y, test_x = None, test_y = None, model = 'lgb', param = None, save_model = None):
        '''
        :param train_x: 训练数据x (最好带列名的Dataframe传入，方便计算特征重要性)
        :param train_y: 训练数据y
        :param test_x,test_y : 测试数据x，测试数据y，在训练的时候同步验证，只有lgb和xgb能这样实现（lgb和xgb最好都传入验证集，可以跟踪训练过程）
        :param save_model : 保存模型，默认False，可以传入文件路径加生成文件名，例如：'model_pkl/gbdt.pkl'
        :return : 返回一个dict，{'clf' : 分类器,'feature_importance':特征重要性}
        '''
        ## 构建模型及传入参数
        self.model = self.__model_pick(model)
        self.__model_build_param(param=param,model_name=model)
        ## 如果train_x不是以DataFrame传入，则raise错误
        self.__data_null_check(train_x)
        ## lgb和xgb可以设置验证集
        if test_x is not None and test_y is not None:
            if self.lgb or self.xgb:
                eval_data = (test_x,test_y)
                train_data = (train_x, train_y)
                ## 评估方式统一采用auc，将早停设置成1000，保证不会早停，然后可以输出评估效果
                clf = self.model.fit(train_x, train_y, early_stopping_rounds= 1000, eval_set=[train_data,eval_data], eval_metric = 'auc')
            else:
                print("只有lgb和xgb才可以训练时同步验证")
                clf = self.model.fit(train_x, train_y)
        else:
            clf = self.model.fit(train_x, train_y)
        ## 输出特征重要性
        feat_name = list(train_x.columns)
        if self.lgb:
            feat_name = clf.booster_.feature_name()
        feat_imp_dic_res = self.__fea_importance(clf, feat_name)
        super_clf = {"clf" : clf, "feature_importance" : feat_imp_dic_res}

        if save_model:
            self.save_model(clf, save_model)
        return super_clf


    def get_feauture_importance(self, train_x, train_y, model = 'all'):
        '''
        :param model: ['lgb','xgb','gbdt','rdf','all']
        :return: 使用默认参数的模型计算特征的重要性，需要输入训练数据（可以多个模型计算特征重要性），只返回特征重要性
        '''
        fea_model = TreeModel()
        def model_fea_importance(fea_model,model_name,train_x,train_y):
            clf_model = fea_model.fit(train_x, train_y, model = model_name)
            clf_model_imp = clf_model['feature_importance']
            return clf_model_imp
        md_feature_imp = {}
        if model in ['lgb','xgb','gbdt','rdf','all']:
            if model == 'all':
                for md_name in ['lgb','xgb','gbdt','rdf']:
                    md_feature_imp[md_name] = model_fea_importance(fea_model = fea_model,model_name = md_name,train_x = train_x,train_y = train_y)
            else:
                md_feature_imp[model] = model_fea_importance(fea_model = fea_model, model_name=model, train_x = train_x,train_y = train_y)
                md_feature_df = pd.DataFrame(md_feature_imp)
            return md_feature_df
        else:
            raise Exception("model必须在[lgb,xgb,gbdt,rdf,all]内")


    def boost_model_evaluate(self, clf):
        '''
        :param clf: 分类器
        :return: 训练数据及验证数据随着迭代变化的auc曲线，只有lgb和xgb才有这个函数，可以使用clf引用该函数
        '''
        if hasattr(clf, 'evals_result_'):
            auc_dict = clf.evals_result_
            train_auc = auc_dict.get(list(auc_dict.keys())[0]).get('auc')
            eva_auc = auc_dict.get(list(auc_dict.keys())[1]).get('auc')
            if len(train_auc) == len(eva_auc):
                length_range = list(range(len(train_auc)))
                ## 根据训练集和验证集的auc画图
                plt.plot(length_range,train_auc,'y--',label = 'train_auc')
                plt.plot(length_range,eva_auc,'b-',label = 'evalidation_auc')
                plt.legend(loc = "upper right")
                plt.title("迭代过程auc变化图")
                plt.show()
        else:
            print("只有lgb和xgb有迭代验证图")


    def cv(self,x,y,model_name = 'lgb',param = None,test_size = 0.3):
        '''
        :param model_name: ['lgb','xgb','gbdt','rdf']
        :param param: 参数
        :param test_size: 验证集比例
        :return: cv会返回验证集比例，验证集auc，验证集ks，分类器
        '''
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=test_size,random_state=0)
        model_cv = TreeModel()
        cv_clf = model_cv.fit(train_x=x_train, train_y=y_train, test_x=x_test, test_y=y_test, model = model_name, param=param)['clf']
        ## 验证ks和auc
        y_pre_proba_ls = cv_clf.predict_proba(x_test)
        y_pre_proba = [p[1] for p in y_pre_proba_ls]
        fpr_,tpr_,_ = roc_curve(y_test,y_pre_proba)
        cv_ks = (tpr_ - fpr_).max()
        cv_auc = auc(fpr_,tpr_)
        cv_res = {"验证集比例" : test_size, "验证集auc" : cv_auc, "验证集ks" : cv_ks, 'clf' : cv_clf}
        return cv_res

    def kfold(self, x, y, k=5, model = 'lgb', param = None):
        '''
        :return: 使用StratifiedKFold进行交叉验证
        '''
        sf_kf = StratifiedKFold(n_splits = k)
        sf_kf_res = []
        i = 0
        for train_idx,test_idx in sf_kf.split(x,y):
            tr_x = x.iloc[train_idx]
            tr_y = y.iloc[train_idx]
            te_x = x.iloc[test_idx]
            te_y = y.iloc[test_idx]
            kf_clf = self.fit(train_x=tr_x, train_y=tr_y, test_x=te_x, test_y=te_y, model = model, param=param)['clf']
            ## 验证ks和auc
            y_pre_proba_ls = kf_clf.predict_proba(te_x)
            y_pre_proba = [p[1] for p in y_pre_proba_ls]
            fpr_, tpr_, _ = roc_curve(te_y, y_pre_proba)
            cv_ks = (tpr_ - fpr_).max()
            cv_auc = auc(fpr_, tpr_)
            i+=1
            sf_kf_res.append({"当前折数": i, "验证集auc": cv_auc, "验证集ks": cv_ks})
        return sf_kf_res

    def save_model(self, clf, filename):
        '''
        :param clf: 需要保存的模型（必须是clf.fit，然后保存fit）
        :param filename: 保存的路径+文件名（pkl类型，如lgb.pkl）
        '''
        joblib.dump(clf, filename)
        print("保存模型{}成功".format(filename))


    def load_model(self, filename):
        '''
        :param filename: 需要加载的模型路径
        :return:
        '''
        model = joblib.load(filename)
        return model

## 模型通用参数（自设）
model_defalt_param = {
    "lgb" : {
            'boosting_type': 'gbdt', 'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5, 'num_leaves': 31,
            'min_child_samples': 20,'min_child_weight': 0.001, 'colsample_bytree': 0.8, 'min_split_gain': 0.0, 'objective': None,
            'subsample': 0.8, 'subsample_freq': 4, 'reg_alpha': 0.1, 'reg_lambda': 0.1, 'silent': True, 'random_state': None
        },
    "xgb" : {
            'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 4, 'min_child_weight': 10, 'subsample': 0.8,
            'colsample_bynode': 0.8, 'colsample_bylevel' : 1,'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
            'base_score': 0.5, 'gamma': 0, 'max_delta_step': 0, 'scale_pos_weight': 1,
            'objective': 'binary:logistic', 'booster': None
        },
    'gbdt' : {
            'n_estimators': 100, 'learning_rate': 0.1, 'subsample': 1.0, 'loss': 'deviance', 'max_features': 0.4,
            'max_depth': 4, 'min_samples_split': 10, 'min_weight_fraction_leaf': 0.0,  'min_samples_leaf': 8,
            'max_leaf_nodes': None, 'min_impurity_split': None, 'criterion': 'friedman_mse','random_state': None,
            'init': None, 'presort': 'auto', 'verbose': 0,'warm_start': False
        },
    'rdf' : {
            'n_estimators': 80, 'max_depth': 5, 'max_features': 0.4, 'min_samples_leaf': 10, 'min_samples_split': 12,
            'max_leaf_nodes': None, 'min_impurity_split': None,  'min_weight_fraction_leaf': 0.0,'criterion': 'gini',
            'class_weight': None, 'bootstrap': True, 'oob_score': True, 'random_state': None, 'verbose': 0, 'warm_start': False
        }
}

## 模型参数解释
model_param_detail = {
    "lgb" : {
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
    },
    "xgb" : {
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
    },
    "gbdt" : {
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
    },
    "rdf" : {
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
}



if __name__ == '__main__':
    data_loc = r'D:\python\jupyter notebook\data\train.csv'
    tr_data = pd.read_csv(data_loc)
    tr_data['Sex'] = tr_data['Sex'].map({'male': 0, 'female': 1})
    tr_data['Age'] = tr_data['Age'].fillna(30)

    feat_ls = ['Pclass', 'Sex', 'Age', 'Parch', 'Fare']
    da_x = tr_data[feat_ls]
    da_y = tr_data['Survived']

    tr_x, te_x = da_x.iloc[:600], da_x.iloc[600:]
    tr_y, te_y = da_y.iloc[:600], da_y.iloc[600:]

    ## K折交叉验证
    # lgb_test = TreeModel()
    # p = lgb_test.kfold(da_x,da_y,model='rdf')

    # 输出通用参数
    # print(lgb_test.model_param_detail['rdf'])

    ## 训练测试
    lgb_test = TreeModel()
    feaimp = lgb_test.fit(tr_x, tr_y, te_x, te_y, model = 'lgb', param= {'max_depth':5}, save_model = 'model_pkl/lgb.pkl')
    # fea2 = lgb_test.fit(tr_x, tr_y, te_x, te_y, model = 'rdf', param= {'max_depth':1})
    # print(fea2['clf'].predict_proba([['1','1','28','3','99.99']]))
    #
    # lgb_test.boost_model_evaluate(feaimp['clf'])
    # lgb_test.boost_model_evaluate(fea2['clf'])
    # print(feaimp['clf'].get_params())
    # print(fea2['clf'].get_params())

    ## 使用四个模型计算特征的重要性
    # imp_test = TreeModel()
    # fea_imp_res = imp_test.get_feauture_importance(tr_x,tr_y,model='all')

    ## cv
    # cv_test = TreeModel()
    # cv_res = cv_test.cv(da_x,da_y,model_name='lgb')
    # print(cv_res['clf'].get_params())
    # cv_test.boost_model_evaluate(cv_res['clf'])
    # print(cv_res)




