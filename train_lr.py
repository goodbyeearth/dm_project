import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.grid_search import GridSearchCV,RandomizedSearchCV
from sklearn import metrics
import pickle

train_address = "/home/train_5000w.txt"

df = pd.read_csv(train_address, sep='\t', header=None, nrows=30000000)

df.columns=['label','depth','position','gender','age',\
           'ctr_adid','ctr_advertiser','ctr_depth','ctr_position','ctr_ratio',\
           'pctr_displayURL','pctr_adid','pctr_advertiserid','pctr_ratio','pctr_query','pctr_keyword', 'pctr_title', 'pctr_description', 'pctr_uid', 'pctr_gender', 'pctr_age',
           'sim_query_keyword', 'sim_query_title', 'sim_query_description', 'sim_keyword_title', 'sim_keyword_description','sim_title_description']
df.loc[:,'gender']=df[['gender']].astype(int)
df.loc[:,'age']=df[['age']].astype(int)


data=df.drop(['label'],axis=1)
retio=0.02
print("train_test_split:"+str(retio))
X_train, X_test, y_train, y_test = train_test_split(data, df['label'], test_size=retio, random_state=101)

quantify_col = ['ctr_adid','ctr_advertiser','pctr_displayURL','pctr_adid','pctr_advertiserid', 'pctr_query',
               'pctr_keyword', 'pctr_title', 'pctr_description', 'pctr_uid',
               'sim_query_keyword', 'sim_query_title', 'sim_query_description', 'sim_keyword_title',
               'sim_keyword_description', 'sim_title_description']
accuracy = 100
for col in quantify_col:
    X_train[col] = X_train[col].apply(lambda x:float(int(x*accuracy)/accuracy))

xgb_model = XGBClassifier()
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['binary:logistic'],
              'learning_rate': [0.1], #so called `eta` value
              'max_depth': [6,10],
              'min_child_weight': [11],
              'silent': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [500], #number of trees, change it to 1000 for better results
              'missing':[-999],
              'seed': [1337],
              'gamma':[10]}

clf = GridSearchCV(xgb_model, parameters, n_jobs=1,
                   cv=3, 
                   scoring='roc_auc',)

clf.fit(X_train, y_train)

proba_Y=clf.predict_proba(X_test)
test_auc = metrics.roc_auc_score(y_test, proba_Y[:,1])#验证集上的auc值
f = open('/home/testFeature/model_3000w', mode='wb')
pickle.dump(clf, f)
f.close()
print("auc={:.5f}\n".format(test_auc))
print(clf.best_estimator_)
#输出最优训练器的精度
print(clf.best_score_)