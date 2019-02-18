import pandas as pd
import pickle


test_set = '/home/test_feature_v2/col26.txt'

df = pd.read_table(test_set, header=None, delim_whitespace=True)
df.columns=['depth','position','gender','age',\
           'ctr_adid','ctr_advertiser','ctr_depth','ctr_position','ctr_ratio',\
           'pctr_displayURL','pctr_adid','pctr_advertiserid','pctr_ratio','pctr_query','pctr_keyword', 'pctr_title', 'pctr_description', 'pctr_uid', 'pctr_gender', 'pctr_age',
           'sim_query_keyword', 'sim_query_title', 'sim_query_description', 'sim_keyword_title', 'sim_keyword_description','sim_title_description',
           ]

df.loc[:,'gender']=df[['gender']].astype(int)
df.loc[:,'age']=df[['age']].astype(int)

f = open('/home/testFeature/model_3000', mode='rb')
clf = pickle.load(f)
f.close()

proba_Y = clf.predict_proba(df)
pd.DataFrame(proba_Y)[1].to_csv('/home/test_feature_v2/test_result.csv',sep='\t',header=False,index=False)