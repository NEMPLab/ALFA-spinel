from pymatgen.core import Composition
from Featurizor import Featurizor
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
import catboost
import xgboost
import optuna
from sklearn.ensemble import GradientBoostingClassifier
from utils import *
from sklearn.model_selection import train_test_split
import json
seed = 42

rf = RandomForestClassifier(n_jobs=-1, random_state=seed)
lgbm = LGBMClassifier(random_state=seed)
ext = ExtraTreesClassifier(random_state=seed, n_jobs=-1)
lr = LogisticRegression(random_state=seed, n_jobs=-1)
xb = xgboost.XGBClassifier(random_state=seed,tree_method='gpu_hist')
cb = catboost.CatBoostClassifier(random_state=seed, thread_count=-1, verbose=0,task_type='GPU')
gb = GradientBoostingClassifier(random_state=seed, verbose=0)

# 读取 JSON 文件
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# 写入 JSON 文件
def write_json_file(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file)

import optuna

def get_rf(X,y,params,iter_num):
    def evaluate_model(trial):
        # 定义超参数搜索范围
        n_estimators = trial.suggest_int('n_estimators', 100, 300)
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
        max_depth = trial.suggest_int('max_depth', 1, 30)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
        max_features = trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2', None])
        bootstrap = trial.suggest_categorical('bootstrap', [True, False])

        # 初始化分类器
        clf = RandomForestClassifier(n_estimators=n_estimators,
                                     criterion=criterion,
                                     max_depth=max_depth,
                                     min_samples_split=min_samples_split,
                                     min_samples_leaf=min_samples_leaf,
                                     max_features=max_features,
                                     bootstrap=bootstrap,
                                    n_jobs=-1,
                                    random_state=seed)

        score,_,train_data = foldn_score(clf,X,y,n=5)
        return score['cs']
    key = 'rf'
    if params['iter_%s'%iter_num] is None or key not in params['iter_%s'%iter_num].keys():
        study = optuna.create_study(direction='maximize')
        study.optimize(evaluate_model, n_trials=200,n_jobs=-1)
        best_params = study.best_params
        params.update({'iter_%s'%iter_num:{key: best_params}})
        write_json_file(params,'./params/params.json')
    else :
        best_params = params['iter_%s'%iter_num][key]
    rf = RandomForestClassifier(n_jobs=-1,random_state=seed,**best_params)
    return rf

import lightgbm as lgb
def get_lgbm(X,y,params,iter_num):
    def evaluate_model(trial):
        # 定义超参数搜索范围
        params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'metric': trial.suggest_categorical('metric', ['binary_logloss', 'binary_error', 'auc', 'precision', 'recall', 'f1']),
            'num_leaves': trial.suggest_int('num_leaves', 10, 1200),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'max_depth': trial.suggest_int('max_depth', 1, 30),
            'verbose': -1
        }


        # 初始化分类器
        clf = LGBMClassifier(andom_state=seed,**params)

        score,_,_ = foldn_score(clf,X,y,n=5)
        return score['cs']
    key = 'lgbm'
    if params['iter_%s'%iter_num] is None or key not in params['iter_%s'%iter_num].keys():
        study = optuna.create_study(direction='maximize')
        study.optimize(evaluate_model, n_trials=200,n_jobs=-1)
        best_params = study.best_params
        params.update({'iter_%s'%iter_num:{key: best_params}})
        write_json_file(params,'./params/params.json')
    else :
        best_params = params['iter_%s'%iter_num][key]
    lgbm = LGBMClassifier(random_state=seed,**best_params)
    return lgbm

def get_lr(X,y,params,iter_num):
    def evaluate_model(trial):
        param_space = {
            'C': trial.suggest_loguniform('C', 0.001, 10),  # 正则化参数
            'penalty': trial.suggest_categorical('penalty', ['l2', 'none']),  # 惩罚项类型
            'l1_ratio': trial.suggest_uniform('l1_ratio', 0, 1),  # ElasticNet混合参数
            'max_iter': trial.suggest_int('max_iter', 0, 2000)  # 最大迭代次数
        }
        clf = LogisticRegression(random_state=seed, n_jobs=-1,**param_space)
        score,_,_ = foldn_score(clf,X,y,n=5)
        return score['cs']
    key = 'lr'
    if params['iter_%s'%iter_num] is None or key not in params['iter_%s'%iter_num].keys():
        study = optuna.create_study(direction='maximize')
        study.optimize(evaluate_model, n_trials=200,n_jobs=-1)
        best_params = study.best_params
        params.update({'iter_%s'%iter_num:{key: best_params}})
        write_json_file(params,'./params/params.json')
    else :
        best_params = params['iter_%s'%iter_num][key]
    lr = LogisticRegression(random_state=seed,n_jobs=-1,**best_params)
    return lr
    
def get_ext(X,y,params,iter_num):
    def evaluate_model(trial):
        n_estimators = trial.suggest_int('n_estimators', 100, 300, step=1)
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
        max_depth = trial.suggest_int('max_depth', 1, 30)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 30)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
        max_features = trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2', None])

        clf = ExtraTreesClassifier(n_estimators=n_estimators,
                                   criterion=criterion,
                                   max_depth=max_depth,
                                   min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf,
                                   max_features=max_features,
                                   random_state=seed,
                                  n_jobs=-1)

        score,_,_ = foldn_score(clf,X,y,n=5)
        return score['cs']

        return score


    key = 'ext'
    if params['iter_%s'%iter_num] is None or key not in params['iter_%s'%iter_num].keys():
        study = optuna.create_study(direction='maximize')
        study.optimize(evaluate_model, n_trials=200,n_jobs=-1)
        best_params = study.best_params
        params.update({'iter_%s'%iter_num:{key: best_params}})
        write_json_file(params,'./params/params.json')
    else :
        best_params = params['iter_%s'%iter_num][key]
    ext = ExtraTreesClassifier(random_state=seed,n_jobs=-1, **best_params)
    return ext

def get_cb(X,y,params,iter_num):
    def evaluate_model(trial):
        param_space = {
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),  # 学习率
            'depth': trial.suggest_int('depth', 1, 16),  # 树的最大深度
            'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 0.1, 10),  # L2 正则化系数
            'iterations': trial.suggest_int('iterations', 50, 300),  # 迭代次数
            'random_strength': trial.suggest_loguniform('random_strength', 0.1, 10)  # 随机强度
        }
        clf = catboost.CatBoostClassifier(random_state=seed,thread_count=-1,task_type='GPU',verbose=0, **param_space)
        score,_,_ = foldn_score(clf, X, y,n=5)
        return score['cs']


    key = 'cb'
    if params['iter_%s'%iter_num] is None or key not in params['iter_%s'%iter_num].keys():
        study = optuna.create_study(direction='maximize')
        study.optimize(evaluate_model, n_trials=200,n_jobs=-1)
        best_params = study.best_params
        params.update({'iter_%s'%iter_num:{key: best_params}})
        write_json_file(params,'./params/params.json')
    else :
        best_params = params['iter_%s'%iter_num][key]
    cb = catboost.CatBoostClassifier(random_state=seed,thread_count=-1,task_type='GPU',verbose=0, **best_params)
    return cb

def get_gb(X,y,params,iter_num):
    def evaluate_model(trial):
        param_space = {
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),  # 学习率
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),  # 弱分类器个数
            'max_depth': trial.suggest_int('max_depth', 1, 30),  # 树的最大深度
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),  # 节点分裂所需的最小样本数
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),  # 叶子节点所需的最小样本数
            'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),  # 每颗树使用的样本比例
        }
        clf = GradientBoostingClassifier(random_state=seed,**param_space)
        score,_,_ = foldn_score(clf, X, y,n=5)
        return score['cs']


    key = 'gb'
    if params['iter_%s'%iter_num] is None or key not in params['iter_%s'%iter_num].keys():
        study = optuna.create_study(direction='maximize')
        study.optimize(evaluate_model, n_trials=200,n_jobs=-1)
        best_params = study.best_params
        params.update({'iter_%s'%iter_num:{key: best_params}})
        write_json_file(params,'./params/params.json')
    else :
        best_params = params['iter_%s'%iter_num][key]
    gb = GradientBoostingClassifier(random_state=seed,**best_params)
    return gb

def get_xb(X,y,params,iter_num):
    def evaluate_model(trial):
        param_space = {
            'eta': trial.suggest_loguniform('eta', 0.001, 0.1),  # 学习率
            'max_depth': trial.suggest_int('max_depth', 2, 30),  # 树的最大深度
            'gamma': trial.suggest_loguniform('gamma', 0.01, 1.0),  # 节点分裂所需的最小增益
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),  # 叶子节点的最小权重
            'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),  # 每颗树使用的样本比例
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),  # 每棵树使用的特征比例
        }
        clf = xgboost.XGBClassifier(random_state=seed,**param_space,tree_method='gpu_hist')
        score,_,_ = fold5_score(clf, X, y,n=5)
        return score['cs']


    key = 'xb'
    if params['iter_%s'%iter_num] is None or key not in params['iter_%s'%iter_num].keys():
        study = optuna.create_study(direction='maximize')
        study.optimize(evaluate_model, n_trials=200,n_jobs=-1)
        best_params = study.best_params
        params.update({'iter_%s'%iter_num:{key: best_params}})
        write_json_file(params,'./params/params.json')
    else :
        best_params = params['iter_%s'%iter_num][key]
    xb = xgboost.XGBClassifier(random_state=seed,**best_params,tree_method='gpu_hist')
    return xb

class Iterate:

    def __init__(self, iter_num,new_spinels_dft=None,need_op=True):
        
        if iter_num == 0:
            need_op = False
            self.spinels = pd.read_csv('./data/input/spinels.csv').reset_index(drop=True)
            self.spinels['formula'] = self.spinels['formula'].map(lambda x: Composition(x).reduced_formula)
        else:
            self.spinels = pd.read_csv('./data/input/spinels_%s.csv'%(iter_num-1)).reset_index(drop=True)
            self.spinels['formula'] = self.spinels['formula'].map(lambda x: Composition(x).reduced_formula)
            new_spinels_dft[r'$y$'] = new_spinels_dft[r'$y$'].map(lambda x: 1 if x else 0)
            new_spinels_dft = new_spinels_dft[['formula', r'$y$']].reset_index(drop=True)
            new_spinels_dft = new_spinels_dft[new_spinels_dft[r'$y$']==1].reset_index(drop=True)
            new_spinels_dft['formula'] = new_spinels_dft['formula'].map(lambda x: Composition(x).reduced_formula)
            self.spinels = pd.concat([self.spinels, new_spinels_dft], axis=0).reset_index(drop=True)
            self.spinels = self.spinels.drop_duplicates('formula').reset_index(drop=True)
            self.params = read_json_file('./params/params.json')
            if 'iter_%s'%iter_num not in self.params.keys():
                self.params['iter_%s'%iter_num] = {}
                write_json_file(self.params,'./params/params.json')
        self.spinels.to_csv('./data/input/spinels_%s.csv'%iter_num)
        self.names = [
            "Random Forest",
            'LGBM',
            "Logistic Regression",
            "ExtraTrees",
            "XGBoost",
            "GradientBoost",
            "CatBoost"
        ]

        self.classifiers = [
            rf,
            lgbm,
            lr,
            ext,
            xb,
            gb,
            cb
        ]
        data = Featurizor(is_normalize=False).featurize(self.spinels,is_structure=False)
        self.X = data.drop(columns=['formula', r'$y$'])
        self.iter_num = iter_num
        self.y = data[r'$y$']
        if need_op:
            self.classifiers = [
            get_rf(self.X,self.y,self.params,iter_num),
            get_lgbm(self.X,self.y,self.params,iter_num),
            get_lr(self.X,self.y,self.params,iter_num),
            get_ext(self.X,self.y,self.params,iter_num),
            get_xb(self.X,self.y,self.params,iter_num),
            get_gb(self.X,self.y,self.params,iter_num),
            get_cb(self.X,self.y,self.params,iter_num)
        ]

    def get_best_models_and_score_df(self):
        X = self.X
        y = self.y
        fold5_df,model,train_data = foldns(self.classifiers, X, y,n=5, names=self.names)
        self.X_train = train_data[0]
        self.y_train = train_data[1]
        max_indices = fold5_df['cs'].nlargest(1).index.tolist()
        max_index = max_indices[0]
        new_row = {'model':'best model','accuracy':fold5_df.loc[max_index,'accuracy'],
                  'f1-score':fold5_df.loc[max_index,'f1-score'],'mcc':fold5_df.loc[max_index,'mcc'],
                   'recall':fold5_df.loc[max_index,'recall'],'cs':fold5_df.loc[max_index,'cs']}
        fold5_df = fold5_df.append(new_row, ignore_index=True)
        return model, fold5_df
    
#     def get_test_score_df(self):
#         def evaluate_test(model,name):
#             _,model,_ = fold5_score(model,self.X,self.y)
#             y_pred = model.predict(self.X_test)
#             y_test = self.y_test
#             return {"model":name,"accuracy":accuracy_score(y_test,y_pred),"f1-score":f1_score(y_test,y_pred),
#                    "mcc":matthews_corrcoef(y_test,y_pred),"cs":(0.5*accuracy_score(y_test,y_pred)+0.5*f1_score(y_test,y_pred))}
#         result = []
#         for model,name in zip(self.classifiers,self.names):
#             result.append(evaluate_test(model,name))
#         result = pd.DataFrame(result)
#         max_index = result['cs'].nlargest(1).index.tolist()[0]
#         new_row = {'model':'best model','accuracy':result.loc[max_index,'accuracy'],
#                   'f1-score':result.loc[max_index,'f1-score'],'mcc':result.loc[max_index,'mcc'],
#                   'cs':result.loc[max_index,'cs']}
#         result = result.append(new_row, ignore_index=True)
#         return result

    def get_new_sampled_spinels_pred(self, model, sample_size=15):
        iter_num = self.iter_num
        if iter_num == 0:
            new_spinels = pd.read_csv('spinels_dft.csv')
        else :
            new_spinels = pd.read_csv('./tmp/spinels_dft_%s.csv'%(iter_num-1))
        new_spinels['formula'] = new_spinels['formula'].map(lambda x: Composition(x).reduced_formula)
        merged = pd.merge(new_spinels, self.spinels['formula'], how='left', indicator=True)

        new_spinels = merged.loc[merged['_merge'] == 'left_only'].drop(columns=['_merge']).reset_index(
            drop=True)
        new_spinels.to_csv('./tmp/spinels_dft_%s.csv'%iter_num,index=False)
        new_spinels[r'$y$'] = 0
        dft = new_spinels['DFT']
        new_spinels_features = Featurizor(is_base=True, is_normalize=False).featurize(new_spinels, is_new_data=True,is_structure=False)
        new_spinels_X = new_spinels_features[self.X.columns]
        new_spinels_features[r'$y$'] = model.predict(new_spinels_X)
        new_spinels_features['DFT'] = dft
        uncertainty = -np.sum(model.predict_proba(new_spinels_X) * np.log2(model.predict_proba(new_spinels_X) + 1e-10), axis=1)
        new_spinels_features['uncertainty'] = uncertainty
        
        if len(new_spinels)<sample_size:
            new_spinels = new_spinels_features[['formula', r'$y$', 'DFT','uncertainty']].reset_index(drop=True)
        else:
            query_indices = np.argsort(uncertainty)[::-1][:sample_size]
            new_spinels = new_spinels_features[['formula', r'$y$', 'DFT','uncertainty']].iloc[query_indices,:].reset_index(drop=True)
        return new_spinels
