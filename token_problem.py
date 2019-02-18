from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

TRAIN_PATH = '/home/data/track2/training.txt'
TRAIN_TOKEN_1 = '/home/data/track2/train_token_1.txt'
TRAIN_TOKEN_4 = '/home/data/track2/train_token_4.txt'
TRAIN_VECTORIZER = '/home/data/track2/vectorizer.txt'
TRAIN_SIMILARITY = '/home/feature/similarity.txt'

TOKEN_1_QUE = '/home/data/track2/queryid_tokensid.txt'
TOKEN_1_KEY = '/home/data/track2/purchasedkeywordid_tokensid.txt'
TOKEN_1_TI = '/home/data/track2/titleid_tokensid.txt'
TOKEN_1_DES = '/home/data/track2/descriptionid_tokensid.txt'

TEST_PATH = '/home/data/test/test.txt'
TEST_TOKEN_1 = '/home/data/test/test_token_1.txt'
TEST_TOKEN_4 = '/home/data/test/test_token_4.txt'
TEST_VECTORIZER = '/home/data/test/vectorizer.txt'
TEST_SIMILARITY = '/home/testFeature/similarity.txt'


def join_token(is_train=True):
    if is_train:
        raw_data_path = TRAIN_PATH
        col_index = [7,8,9,10]
        token_4_path = TRAIN_TOKEN_4
        token_1_path = TRAIN_TOKEN_1
    else:
        raw_data_path = TEST_PATH
        col_index = [5,6,7,8]
        token_4_path = TEST_TOKEN_4
        token_1_path = TEST_TOKEN_1

    df_main = pd.read_csv(raw_data_path, sep='\t', usecols=col_index, names=['queryID','keywordID',
                       'titleID','descriptionID'], header=None)
    df_query = pd.read_csv(TOKEN_1_QUE, sep='\t', names=['queryID','token1'],header=None)
    df_keyword = pd.read_csv(TOKEN_1_KEY, sep='\t', names=['keywordID','token2'],header=None)
    df_title = pd.read_csv(TOKEN_1_TI, sep='\t', names=['titleID', 'token3'], header=None)
    df_description = pd.read_csv(TOKEN_1_DES, sep='\t', names=['descriptionID', 'token4'], header=None)
    df_main = df_main.join(df_query.set_index('queryID')['token1'], on='queryID')
    df_main = df_main.join(df_keyword.set_index('keywordID')['token2'], on='keywordID')
    df_main = df_main.join(df_title.set_index('titleID')['token3'], on='titleID')
    df_main = df_main.join(df_description.set_index('descriptionID')['token4'], on='descriptionID')
    df_main[['token1','token2','token3','token4']].to_csv(token_4_path, sep='\t', index=False, header=False)

    df_query[['token1']].to_csv(token_1_path, mode='a', sep='\t', index=False, header=False)
    df_keyword[['token2']].to_csv(token_1_path, mode='a', sep='\t', index=False, header=False)
    df_title[['token3']].to_csv(token_1_path, mode='a', sep='\t', index=False, header=False)
    df_description[['token4']].to_csv(token_1_path, mode='a', sep='\t', index=False, header=False)
    print("token joining completed!")


def get_vectorizer(is_train=True):
    print("getting vectorizer ...")
    if is_train:
        token_1_path = TRAIN_TOKEN_1
        vectorizer_path = TRAIN_VECTORIZER
    else:
        token_1_path = TEST_TOKEN_1
        vectorizer_path = TEST_VECTORIZER

    data = pd.read_csv(token_1_path, sep='\t', header=None,names=['token'])
    tfidf_vectorizer = TfidfVectorizer(lowercase=False)
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['token'])
    f = open(vectorizer_path, mode='wb')
    pickle.dump(tfidf_vectorizer, f)
    f.close()
    print('vectorizer got!')


def get_similarity(is_training=True):
    if is_training:
        token_4_path = TRAIN_TOKEN_4
        vectorizer_path = TRAIN_VECTORIZER
        similarity_path = TRAIN_SIMILARITY
    else:
        token_4_path = TEST_TOKEN_4
        vectorizer_path = TEST_VECTORIZER
        similarity_path = TEST_SIMILARITY

    chunker = pd.read_csv(token_4_path, sep='\t',names=['token1','token2','token3','token4'],
                       chunksize=500, header=None)
    f = open(vectorizer_path, mode='rb')
    vectorizer = pickle.load(f)
    f.close()
    i = 0

    for c in chunker:
        m1 = c['token1']
        m2 = c['token2']
        m3 = c['token3']
        m4 = c['token4']

        v1 = vectorizer.transform(m1)
        v2 = vectorizer.transform(m2)
        v3 = vectorizer.transform(m3)
        v4 = vectorizer.transform(m4)
        cos = cosine_similarity(v1, v2, dense_output=False).diagonal()
        df = pd.DataFrame(cos)
        cos = cosine_similarity(v1, v3, dense_output=False).diagonal()
        df = df.join(pd.Series(cos, name='2'))
        cos = cosine_similarity(v1, v4, dense_output=False).diagonal()
        df = df.join(pd.Series(cos, name='3'))
        cos = cosine_similarity(v2, v3, dense_output=False).diagonal()
        df = df.join(pd.Series(cos, name='4'))
        cos = cosine_similarity(v2, v4, dense_output=False).diagonal()
        df = df.join(pd.Series(cos, name='5'))
        cos = cosine_similarity(v3, v4, dense_output=False).diagonal()
        df = df.join(pd.Series(cos, name='6'))

        df.to_csv(similarity_path, sep='\t', mode='a', index=False, header=False)
        print(i,' th chunk was completed!')
        i += 1


if __name__ == '__main__':
    is_train = False
    join_token(is_train)
    get_vectorizer(is_train)
    get_similarity(is_train)