import pandas as pd
import os
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

TRAIN_PATH = '/home/data/track2/train.txt'
TRAIN_SIM = '/home/feature/similarity.txt'
TRAIN_LDPGA = '/home/feature/ldpga.txt'
TRAIN_COL27 = '/home/feature/col27.txt'
TRAIN_CP = '/home/feature/merged_cp.txt'
TRAIN_CPS = '/home/feature/merged_cps.txt'

TEST_PATH = '/home/data/test/test_ga.txt'
TEST_SIM = '/home/testFeature/similarity.txt'
TEST_DPGA = '/home/testFeature/dpga.txt'
TEST_CP = '/home/testFeature/cp.txt'
TEST_CPS = '/home/testFeature/cps.txt'
TEST_COL26 = '/home/testFeature/col26.txt'


COLUMNS = ['displayURL', 'adID', 'advertiserID', 'depth', 'position','queryID',
           'keywordID', 'titleID', 'descriptionID', 'userID', 'gender', 'age']
FEATURE_SAVE = ['ctr'+str(i) for i in range(1,6)] + ['pctr'+str(j) for j in range(1,12)]


def merge_cps(is_train=True):
    if is_train:
        raw_data_path = TRAIN_PATH
        sim_path = TRAIN_SIM
        cp_path = TRAIN_CP
        cps_path = TRAIN_CPS
        col_index = [x for x in range(2, 14)]
    else:
        raw_data_path = TEST_PATH
        sim_path = TEST_SIM
        cp_path = TEST_CP
        cps_path = TEST_CPS
        col_index = [x for x in range(12)]

    round = 0
    chunker_mains = pd.read_csv(raw_data_path, sep='\t', header=None, chunksize=5000000,
                          usecols=col_index, names=COLUMNS)
    chunker_sims = pd.read_csv(sim_path, sep='\t', header=None, chunksize=5000000,
                         names=['sim' + str(n) for n in range(1, 7)])

    df_ctr_ad = pd.read_csv(CTR_AD, sep='\t', header=None, names=['adID','ctr1'])
    df_ctr_er = pd.read_csv(CTR_ER, sep='\t', header=None, names=['advertiserID', 'ctr2'])
    df_ctr_dep = pd.read_csv(CTR_DEP, sep='\t', header=None, names=['depth', 'ctr3'])
    df_ctr_pos = pd.read_csv(CTR_POS, sep='\t', header=None, names=['position', 'ctr4'])
    df_ctr_ratio = pd.read_csv(CTR_RATIO, sep='\t', header=None, names=['ratio', 'ctr5'])

    df_pctr_dis = pd.read_csv(PCTR_DIS, sep='\t', header=None, names=['displayURL','pctr1'])
    df_pctr_ad = pd.read_csv(PCTR_AD, sep='\t', header=None, names=['adID','pctr2'])
    df_pctr_er = pd.read_csv(PCTR_ER, sep='\t', header=None, names=['advertiserID','pctr3'])
    df_pctr_ratio = pd.read_csv(PCTR_RATIO, sep='\t', header=None, names=['ratio','pctr4'])
    df_pctr_query = pd.read_csv(PCTR_QUE, sep='\t', header=None, names=['queryID','pctr5'])
    df_pctr_keyword = pd.read_csv(PCTR_KEY, sep='\t', header=None, names=['keywordID','pctr6'])
    df_pctr_title = pd.read_csv(PCTR_TI, sep='\t', header=None, names=['titleID','pctr7'])
    df_pctr_description = pd.read_csv(PCTR_DES, sep='\t', header=None, names=['descriptionID','pctr8'])
    df_pctr_uid = pd.read_csv(PCTR_UID, sep='\t', header=None, names=['userID','pctr9'])
    df_pctr_gender = pd.read_csv(PCTR_GEN, sep='\t', header=None, names=['gender','pctr10'])
    df_pctr_age = pd.read_csv(PCTR_AGE, sep='\t', header=None, names=['age','pctr11'])

    for df_main, df_sim in zip(chunker_mains, chunker_sims):
        df_main['ratio'] = (df_main['depth'] - df_main['position']) / df_main['depth']

        df_main = df_main.join(df_ctr_ad.set_index('adID')['ctr1'], on='adID')
        df_main['ctr1'] = df_main['ctr1'].fillna(0.034703)
        df_main = df_main.join(df_ctr_er.set_index('advertiserID')['ctr2'], on='advertiserID')
        df_main['ctr2'] = df_main['ctr2'].fillna(0.034186)
        df_main = df_main.join(df_ctr_dep.set_index('depth')['ctr3'], on='depth')
        df_main['ctr3'] = df_main['ctr3'].fillna(0.035242)
        df_main = df_main.join(df_ctr_pos.set_index('position')['ctr4'], on='position')
        df_main['ctr4'] = df_main['ctr4'].fillna(0.034804)
        df_main = df_main.join(df_ctr_ratio.set_index('ratio')['ctr5'], on='ratio')
        df_main['ctr5'] = df_main['ctr5'].fillna(0.035203)

        df_main = df_main.join(df_pctr_dis.set_index('displayURL')['pctr1'], on='displayURL')
        df_main['pctr1'] = df_main['pctr1'].fillna(0.034447)
        df_main = df_main.join(df_pctr_ad.set_index('adID')['pctr2'], on='adID')
        df_main['pctr2'] = df_main['pctr2'].fillna(0.035702)
        df_main = df_main.join(df_pctr_er.set_index('advertiserID')['pctr3'], on='advertiserID')
        df_main['pctr3'] = df_main['pctr3'].fillna(0.034295)
        df_main = df_main.join(df_pctr_ratio.set_index('ratio')['pctr4'], on='ratio')
        df_main['pctr4'] = df_main['pctr4'].fillna(0.035203)
        df_main = df_main.join(df_pctr_query.set_index('queryID')['pctr5'], on='queryID')
        df_main['pctr5'] = df_main['pctr5'].fillna(0.045717)
        df_main = df_main.join(df_pctr_keyword.set_index('keywordID')['pctr6'], on='keywordID')
        df_main['pctr6'] = df_main['pctr6'].fillna(0.035767)
        df_main = df_main.join(df_pctr_title.set_index('titleID')['pctr7'], on='titleID')
        df_main['pctr7'] = df_main['pctr7'].fillna(0.037443)
        df_main = df_main.join(df_pctr_description.set_index('descriptionID')['pctr8'], on='descriptionID')
        df_main['pctr8'] = df_main['pctr8'].fillna(0.037037)
        df_main = df_main.join(df_pctr_uid.set_index('userID')['pctr9'], on='userID')
        df_main['pctr9'] = df_main['pctr9'].fillna(0.040579)
        df_main = df_main.join(df_pctr_gender.set_index('gender')['pctr10'], on='gender')
        df_main['pctr10'] = df_main['pctr10'].fillna(0.034897)
        df_main = df_main.join(df_pctr_age.set_index('age')['pctr11'], on='age')
        df_main['pctr11'] = df_main['pctr11'].fillna(0.035412)

        df_main[FEATURE_SAVE].to_csv(cp_path, sep='\t', mode='a', index=False, header=False)
        print(round, ' th chunk completed!')
        round += 1

    print("Ctr and pctr merged! Now paste the similarity.txt")
    os.system('paste ' + cp_path + ' ' + sim_path + ' > ' + cps_path)
    print("Pasting completed!")
    # os.system('7z a /home/feature/gps.7z '+ cps_path)


# merge click, depth, position, ratio, gender, age
def merge_ldpga(is_train=True):
    if is_train:
        col_index = [0, 5, 6, 12, 13]
        col_name = ['label', 'depth', 'position', 'gender', 'age']
        d1_path = TRAIN_PATH
        file1 = TRAIN_LDPGA
        file2 = TRAIN_CPS
        file3 = TRAIN_COL27
    else:
        col_index = [3, 4, 10, 11]
        col_name = ['depth', 'position', 'gender', 'age']
        d1_path = TEST_PATH
        file1 = TEST_DPGA
        file2 = TEST_CPS
        file3 = TEST_COL26
        
    d1 = pd.read_csv(d1_path, sep='\t', header=None, usecols=col_index, names=col_name)
    if is_train:
        d1['label'] = d1['label'].apply(click_to_label)
    print("Now writing (L)DPGA ...")
    d1.to_csv(file1, sep='\t', header=False, index=False)
    print("Writing completed! Now merge all")
    os.system('paste ' + file1 + ' ' + file2 + ' > ' + file3)
    

def click_to_label(click):
    if click >= 1:
        return 1
    return click


if __name__ == '__main__':
    is_train = False
    # merge_cps(is_train)
    merge_ldpga(is_train)