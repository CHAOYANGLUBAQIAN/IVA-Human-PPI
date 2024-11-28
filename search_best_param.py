# 导入所需的库
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import numpy as np
import warnings
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier
import lightgbm as lgb
from sklearn.naive_bayes import MultinomialNB


warnings.filterwarnings("ignore")
# 加载数据集
input_file1 = 'data/features/CT.csv'
df = pd.read_csv(input_file1, header=0, encoding='utf-8')

# 提取特征和标签
X = df.iloc[:, 1:-1] #取属性值 不包含第一列的ID和最后一列的标签
y = df.iloc[:,-1]  #最后一列为标签
# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
# # 实例化XGBoost分类器






#adaboost model
base_classifier = DecisionTreeClassifier()
model = AdaBoostClassifier(base_classifier)
param_grid = {
    'n_estimators': list(np.arange(50,500,10)),
    'learning_rate':list(np.arange(0.01,0.2,0.01))
    'algorithm':['SAMME', 'SAMME.R']

}





#ExtraTrees model
#model = ExtraTreesClassifier()
#param_grid = {
#    'n_estimators': list(np.arange(50,500,10)),
#    'criterion':['gini', 'entropy']
#    'max_depth':[None, 2, 4, 6, 8, 10]
#    'min_samples_split':list(np.arange(2, 10, 1))
#    'min_samples_leaf':list(np.arange(1, 10, 1))
#    'max_features':['sqrt', 'log2']
#}



#Random Forest model
#model = RandomForestClassifier()
#param_grid = {
#    'n_estimators': list(np.arange(50,500,10)),
#    'max_depth':[None, 2, 4, 6, 8, 10]
#    'min_samples_split':list(np.arange(2, 10, 1))
#    'min_samples_leaf':list(np.arange(1, 10, 1))
#    'max_features':['sqrt', 'log2']
#    'criterion':['gini', 'entropy']
#}



#svm model
#model = svm.SVC(kernel='rbf')
#param_grid = {
#    'C': [0.1, 1, 10, 100, 200]
#    'gamma':[0.01, 0.1, 1, 5, 10, 50, 100]
#    'kernel':['linear', 'rbf', 'poly']
#}



#XGBoost model
model =  XGBClassifier()
#param_grid = {
#    'n_estimators': list(np.arange(50,500,10)),
#    'learning_rate':list(np.arange(0.01,0.2,0.01))
#    'max_depth':list(np.arange(3, 10, 1))
#    'min_child_weidht':list(np.arange(1, 10, 1))
#    'gamma':list(np.arange(0.0, 0.3, 0.1))
#    'reg_alpha':list(np.arange(0.0, 1, 0.1))
#    'reg_lambda':list(np.arange(0.0, 1, 0.1))
#}





#LogisticRegression model
model = LogisticRegression()
#param_grid = {
#    'C': [0.1, 1, 10, 100, 200]
#    'penalty':['l1', 'l2']
#}

# 使用网格搜索选择最佳参数
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kf,scoring='precision')
grid_search.fit(X_train, y_train)



# 输出最佳参数和得分
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)
