import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
from scoreAUC import scoreClickAUC

train_address = "/home/feature/col27.txt"
test_address = "/home/testFeature/col26.txt"
test_label_address = "/home/testFeature/KDD_solution.csv"

train_df=pd.read_table(train_address, header=None,delim_whitespace=True,skiprows=110000000)
test_df=pd.read_table(test_address, header=None,delim_whitespace=True)
test_lable=pd.read_csv(test_label_address)

train_df.columns=['label','depth','position','gender','age',\
           'ctr_adid','ctr_advertiser','ctr_depth','ctr_position','ctr_ratio',\
           'pctr_displayURL','pctr_adid','pctr_advertiserid','pctr_ratio','pctr_query','pctr_keyword', 'pctr_title', 'pctr_description', 'pctr_uid', 'pctr_gender', 'pctr_age',
           'sim_query_keyword', 'sim_query_title', 'sim_query_description', 'sim_keyword_title', 'sim_keyword_description','sim_title_description',
           ]
train_df.loc[:,'gender']=train_df[['gender']].astype(int)
train_df.loc[:,'age']=train_df[['age']].astype(int)


test_df.columns=['depth','position','gender','age',\
           'ctr_adid','ctr_advertiser','ctr_depth','ctr_position','ctr_ratio',\
           'pctr_displayURL','pctr_adid','pctr_advertiserid','pctr_ratio','pctr_query','pctr_keyword', 'pctr_title', 'pctr_description', 'pctr_uid', 'pctr_gender', 'pctr_age',
           'sim_query_keyword', 'sim_query_title', 'sim_query_description', 'sim_keyword_title', 'sim_keyword_description','sim_title_description',
           ]
test_df.loc[:,'gender']=test_df[['gender']].astype(int)
test_df.loc[:,'age']=test_df[['age']].astype(int)


 # 相对位置
def Qu(x): # 相对位置离散化 越接近首位值越低
    if x>=1:
        return 1
    elif x>= 0.6 :
        return 2
    elif x>=0.5 :
        return 3
    elif x>=0.3 :
        return 4
    else:
        return 5


def rp(trian_data):
    tmp= (trian_data['depth']-trian_data['position']) 
    tmp = (tmp+1) 
    tmp = tmp/trian_data['depth']
    trian_data['RP']=tmp # relative pos
    tmp=trian_data[['RP']].applymap(Qu)
    trian_data['RP']=tmp


rp(test_df)
rp(train_df)


def onehot(train_data,test_data):
    train_data=pd.get_dummies(train_data,columns=['gender'])
    train_data=pd.get_dummies(train_data,columns=['age'])
    train_data=pd.get_dummies(train_data,columns=['RP'])
    test_data=pd.get_dummies(test_data,columns=['gender'])
    test_data=pd.get_dummies(test_data,columns=['age'])
    test_data=pd.get_dummies(test_data,columns=['RP'])


onehot(train_df,test_df)


lr = LogisticRegression(
            solver='sag',
    class_weight='balanced',
    max_iter=10000
)
lr.fit(train_df.drop(['label'],axis=1),train_df['label'])

f = open('/home/testFeature/lr_3000w', mode='wb')
pickle.dump(lr, f)
f.close()

# 输出结果
proba_Y = lr.predict_proba(test_df)

test_auc=scoreClickAUC(test_lable['clicks'],test_lable['impressions'],proba_Y[:,1])
print(test_auc)
pd.DataFrame(proba_Y)[1].to_csv('/home/testFeature/result_3000w.csv',header=False,index=False)