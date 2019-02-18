import pandas as pd
import random
TRAIN_PATH = '/home/data/track2/training.txt'
TRAIN_UGA = '/home/feature/user_gender_age.txt'

USER_PATH = '/home/data/track2/userid_profile.txt'

TEST_PATH = '/home/data/test/test.txt'
TEST_UGA = '/home/data/test/uga.txt'

CTR_AD = '/home/feature/ctr_ad.txt'
CTR_ER = '/home/feature/ctr_er.txt'
CTR_DEP = '/home/feature/ctr_dep.txt'
CTR_POS = '/home/feature/ctr_pos.txt'
CTR_RATIO = '/home/feature/ctr_ratio.txt'

PCTR_AD = '/home/feature/pctr_ad.txt'
PCTR_ER = '/home/feature/pctr_er.txt'
PCTR_QUE = '/home/feature/pctr_que.txt'
PCTR_KEY = '/home/feature/pctr_key.txt'
PCTR_TI = '/home/feature/pctr_ti.txt'
PCTR_DES = '/home/feature/pctr_des.txt'
PCTR_UID = '/home/feature/pctr_uid.txt'
PCTR_DIS = '/home/feature/pctr_dis.txt'
PCTR_GEN = '/home/feature/pctr_gen.txt'
PCTR_AGE = '/home/feature/pctr_age.txt'
PCTR_RATIO = '/home/feature/pctr_ratio.txt'


COLUMNS = ['click','impression', 'displayURL', 'adID', 'advertiserID','depth',
            'position','queryID','keywordID','titleID','descriptionID','userID']


def actr_feature():
    data = pd.read_csv(TRAIN_PATH, sep='\t', usecols=[0, 1, 3, 4, 5, 6], names=['click', 'impression',
                        'adID', 'advertiserID', 'depth', 'position'], header=None)

    temp = data.groupby('adID')['click', 'impression'].sum()    # the index is adID
    temp['actr_adID'] = temp['click']/temp['impression']
    temp['actr_adID'].to_csv(CTR_AD, sep='\t', index=True, header=False)

    temp = data.groupby('advertiserID')['click', 'impression'].sum()
    temp['actr_advertiserID'] = temp['click'] / temp['impression']
    temp['actr_advertiserID'].to_csv(CTR_ER, sep='\t', index=True, header=False)

    temp = data.groupby('depth')['click', 'impression'].sum()
    temp['actr_depth'] = temp['click'] / temp['impression']
    temp['actr_depth'].to_csv(CTR_DEP, sep='\t', index=True, header=False)

    temp = data.groupby('position')['click', 'impression'].sum()
    temp['actr_position'] = temp['click'] / temp['impression']
    temp['actr_position'].to_csv(CTR_POS, sep='\t', index=True, header=False)

    data['ratio'] = (data['depth']-data['position'])/data['depth']
    temp = data.groupby('ratio')['click', 'impression'].sum()
    temp['actr_ratio'] = temp['click']/temp['impression']
    temp['actr_ratio'].to_csv(CTR_RATIO, sep='\t', index=True, header=False)
    alpha = 0.05
    beta = 75
    temp['pctr_ratio'] = (temp['click'] + alpha * beta) / (temp['impression'] + beta)
    temp['pctr_ratio'].to_csv(PCTR_RATIO, sep='\t', index=True, header=False)


def pctr_feature_v1():
    data = pd.read_csv(TRAIN_PATH, sep='\t', usecols=[0, 1, 3, 4, 7, 8], names=['click', 'impression',
                        'adID', 'advertiserID', 'queryID', 'keywordID'], header=None)
    alpha = 0.05
    beta = 75

    temp = data.groupby('adID')['click', 'impression'].sum()
    temp['pctr_adID'] = (temp['click'] + alpha * beta) / (temp['impression'] + beta)
    temp['pctr_adID'].to_csv(PCTR_AD, sep='\t', index=True, header=False)

    temp = data.groupby('advertiserID')['click', 'impression'].sum()
    temp['pctr_advertiserID'] = (temp['click'] + alpha * beta) / (temp['impression'] + beta)
    temp['pctr_advertiserID'].to_csv(PCTR_ER, sep='\t', index=True, header=False)

    temp = data.groupby('queryID')['click', 'impression'].sum()
    temp['pctr_queryID'] = (temp['click'] + alpha * beta) / (temp['impression'] + beta)
    temp['pctr_queryID'].to_csv(PCTR_QUE, sep='\t', index=True, header=False)

    temp = data.groupby('keywordID')['click', 'impression'].sum()
    temp['pctr_keywordID'] = (temp['click'] + alpha * beta) / (temp['impression'] + beta)
    temp['pctr_keywordID'].to_csv(PCTR_KEY, sep='\t', index=True, header=False)


def pctr_feature_v2():
    data = pd.read_csv(TRAIN_PATH, sep='\t', usecols=[0, 1, 2, 9, 10, 11], names=['click', 'impression',
                            'displayURL', 'titleID', 'descriptionID', 'userID'], header=None)
    alpha = 0.05
    beta = 75

    temp = data.groupby('titleID')['click', 'impression'].sum()
    temp['pctr_titleID'] = (temp['click'] + alpha * beta) / (temp['impression'] + beta)
    temp['pctr_titleID'].to_csv(PCTR_TI, sep='\t', index=True, header=False)

    temp = data.groupby('descriptionID')['click', 'impression'].sum()
    temp['pctr_descriptionID'] = (temp['click'] + alpha * beta) / (temp['impression'] + beta)
    temp['pctr_descriptionID'].to_csv(PCTR_DES, sep='\t', index=True, header=False)

    temp = data.groupby('userID')['click', 'impression'].sum()
    temp['pctr_userID'] = (temp['click'] + alpha * beta) / (temp['impression'] + beta)
    temp['pctr_userID'].to_csv(PCTR_UID, sep='\t', index=True, header=False)

    temp = data.groupby('displayURL')['click', 'impression'].sum()
    temp['pctr_displayURL'] = (temp['click'] + alpha * beta) / (temp['impression'] + beta)
    temp['pctr_displayURL'].to_csv(PCTR_DIS, sep='\t', index=True, header=False)


def pctr_feature_v3():
    data_c_i = pd.read_csv(TRAIN_PATH, sep='\t', usecols=[0, 1],names=['click','impression'],header=None)
    data_g_a = pd.read_csv(TRAIN_UGA, sep='\t', usecols=[1,2],names=['gender','age'], header=None)
    if len(data_c_i) == len(data_g_a):
        data_c_i = data_c_i.join(data_g_a)
        alpha = 0.05
        beta = 75

        temp = data_c_i.groupby('gender')['click', 'impression'].sum()
        temp['pctr_gender'] = (temp['click'] + alpha*beta)/(temp['impression']+beta)
        temp['pctr_gender'].to_csv(PCTR_GEN, sep='\t', index=True, header=False)

        temp = data_c_i.groupby('age')['click', 'impression'].sum()
        temp['pctr_age'] = (temp['click'] + alpha * beta) / (temp['impression'] + beta)
        temp['pctr_age'].to_csv(PCTR_AGE, sep='\t', index=True, header=False)

    else:
        print('the length is not equivalent!')


def get_user(is_train=True):
    if is_train:
        raw_data_path = TRAIN_PATH
        col_name = COLUMNS
        uga_path = TRAIN_UGA
    else:
        raw_data_path = TEST_PATH
        uga_path = TEST_UGA
        col_name = ['displayURL', 'adID', 'advertiserID','depth','position',
                    'queryID','keywordID','titleID','descriptionID','userID']

    chunker = pd.read_csv(raw_data_path, sep='\t', chunksize=5000000, names=col_name, header=None)
    user = pd.read_csv(USER_PATH, sep='\t', names=['userID', 'gender', 'age'], header=None)
    choice_list = [3,3,3,3,3,3,3,4,4,4,4,4,2,2,2,2,2,5,5,5,5,1,1,6]

    for c in chunker:
        uid = pd.DataFrame(c['userID'])
        uid = uid.join(user.set_index('userID')[['gender', 'age']], on='userID')
        uid['gender'] = uid['gender'].fillna(random.randint(1,2))
        uid['age'] = uid['age'].fillna(random.choice(choice_list))
        uid[['gender','age']].to_csv(uga_path, sep='\t',mode='a', index=False, header=False)


if __name__ == '__main__':
    get_user(False)



