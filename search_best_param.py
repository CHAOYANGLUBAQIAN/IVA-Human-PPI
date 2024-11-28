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
warnings.filterwarnings("ignore")
# 加载数据集
input_file1 = 'data/features/合/基于度的差异负采样/CT.csv'
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
}


#svm model
#model = svm.SVC(kernel='rbf')
#param_grid = {
#    'n_estimators': list(np.arange(50,500,10)),
#    'learning_rate':list(np.arange(0.01,0.2,0.01))
#}



#KNN model
model = KNeighborsClassifier()
#param_grid = {
#    'n_estimators': list(np.arange(50,500,10)),
#    'learning_rate':list(np.arange(0.01,0.2,0.01))
#}


#LogisticRegression model
model = LogisticRegression()
#param_grid = {
#    'n_estimators': list(np.arange(50,500,10)),
#    'learning_rate':list(np.arange(0.01,0.2,0.01))
#}

# 使用网格搜索选择最佳参数
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kf,scoring='precision')
grid_search.fit(X_train, y_train)



# 输出最佳参数和得分
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)
