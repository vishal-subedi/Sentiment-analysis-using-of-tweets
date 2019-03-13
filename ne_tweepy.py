from nltk.tokenize import TweetTokenizer, word_tokenize, sent_tokenize
tt = TweetTokenizer()
import pandas as pd
import tweepy
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import re
import numpy as np
import xgboost
import scipy as sp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from textblob import TextBlob, Word
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import f1_score as f1, silhouette_score as sil
from sklearn.decomposition import PCA
from sklearn.externals import joblib
import string
from keras.models import Sequential
from keras.layers import Dense
from sklearn.cluster import KMeans
stop = stopwords.words('english')
ss = StandardScaler()
mabs = MaxAbsScaler()
pca = PCA(100)
lemmatizer = WordNetLemmatizer()


#consumer key, consumer secret, access token, access secret.
consumer_key = 'vYvcVMYuG5E6zkCKE1Bv20OmP'
consumer_secret = 'IeQdoaHIyshQ5RKXVgjeIG2bYRrn1M93McI6K4tm1WjB9fqaP6'
access_token = '147465780-2XJcemtPIhY5yvHJtEkztDP2qqxfYfBGX0ZpG61i'
access_token_secret = 'oGA3F5njwq0AHvZRBjcBYZAMwJjS831N3bqU5NEy2zY9i'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
   
api = tweepy.API(auth, wait_on_rate_limit = True)



time = []
name = []
screen_name = []
post = []
location = []

for tweet in tweepy.Cursor(api.search, q = ('Tuberculosis'), tweet_mode = 'extended', lang = 'en').items(1000):
    Name = tweet.author.name
    Screen_name = tweet.author.screen_name
    Time = tweet.created_at
    Tweet = tweet.full_text
    Location = tweet.user.location
    name.append(Name)
    screen_name.append(Screen_name)
    time.append(Time)
    post.append(Tweet)
    location.append(Location)



df = pd.DataFrame(columns = ['Time', 'Tweet', 'Location', 'Screen_Name', 'Name'])
df.head()



for i in range(len(post)):
    df = df.append({'Time' : time[i], 'Tweet' : post[i], 'Location' : location[i], 'Screen_Name' : screen_name[i], 'Name' : name[i]}, ignore_index = True)




df.to_csv('tb_day_3_test.csv')


df1 = pd.read_csv('Desktop/Datasets/Disease/mal_day_2.csv', header = 0, index_col = 0)



df2 = pd.read_csv('Desktop/Datasets/Disease/den_day_2.csv', header = 0, index_col = 0)



df3 = pd.read_csv('Desktop/Datasets/Disease/typ_day_2.csv', header = 0, index_col = 0)



df4 = pd.read_csv('Desktop/Datasets/Disease/can_day_2.csv', header = 0, index_col = 0)



df5 = pd.read_csv('Desktop/Datasets/Disease/tb_day_2.csv', header = 0, index_col = 0)



df1n = df1.loc[:, 'Tweet']


df2n = df2.loc[:, 'Tweet']


df3n = df3.loc[:, 'Tweet']


df4n = df4.loc[:, 'Tweet']


df5n = df5.loc[:, 'Tweet']


l1 = []
l2 = []
l3 = []
l4 = []
l5 = []



def labelize(arr, dataframe, label):
    for i in range(dataframe.shape[0]):
        arr.append(label)
    return 0



labelize(l1, df1n, 0)


labelize(l2, df2n, 1)



labelize(l3, df3n, 2)


labelize(l4, df4n, 3)


labelize(l5, df5n, 4)



ser1 = pd.Series(l1)



ser2 = pd.Series(l2)


ser3 = pd.Series(l3)


ser4 = pd.Series(l4)


ser5 = pd.Series(l5)


frame1 = [df1n, ser1]
df_mal = pd.concat(frame1, axis = 1)


frame2 = [df2n, ser2]
df_den = pd.concat(frame2, axis = 1)


frame3 = [df3n, ser3]
df_typ = pd.concat(frame3, axis = 1)



frame4 = [df4n, ser4]
df_can = pd.concat(frame4, axis = 1)


frame5 = [df5n, ser5]
df_tb = pd.concat(frame5, axis = 1)



def column_name(dataframe):
    dataframe.columns = ['Tweet', 'Label']



column_name(df_mal)


column_name(df_den)



column_name(df_can)



column_name(df_typ)



column_name(df_tb)



df_mal.head()



df_den.head()



df_typ.head()



df_can.head()


df_tb.head()


df_disease = pd.concat([df_mal, df_den, df_typ, df_can, df_tb], axis = 0)


# mal -> 0
# dengue -> 1
# typhoid -> 2
# cancer -> 3
# tuberculosis -> 4


df_disease.head(df_disease.shape[0])



df_disease = df_disease.sample(frac = 1).reset_index(drop = True)



df_disease.head(df_disease.shape[0])



for i in range(df_disease.shape[0]):
    df_disease.iloc[i, 0] = df_disease.iloc[i, 0].lower()

    
    
def remove_punc(sent):
    table = str.maketrans({key: None for key in string.punctuation})
    return sent.translate(table)           



for i in range(df_disease.shape[0]):
    df_disease.iloc[i, 0] = remove_punc(df_disease.iloc[i, 0])
    
    




def clean_tweet(tweet): 
        ''' 
        Utility function to clean tweet text by removing links, special characters 
        using simple regex statements. 
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())


for i in range(df_disease.shape[0]):
    df_disease.loc[i, 'Tweet'] = clean_tweet(df_disease.loc[i, 'Tweet'])
    
    
    
    
def tokenize_and_stem(sent):
    stemmer = SnowballStemmer("english")
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(sent)
    filtered = [w for w in words if w not in stop_words]
    stems = [stemmer.stem(t) for t in filtered]
    return stems



df_disease.head(df_disease.shape[0])



df_disease.iloc[0,0]



tfidf = TfidfVectorizer(max_features = 100, use_idf = True, stop_words = 'english', tokenizer = tokenize_and_stem)
tfidf_matrix = tfidf.fit_transform(df_disease.iloc[:, 0])   

terms = []
terms = tfidf.get_feature_names()


wordvec_df = tfidf_matrix.toarray()

wordvec_df = pd.DataFrame(wordvec_df)

df_final = pd.concat([df_disease, wordvec_df], axis = 1)





polarity = pd.Series(df_final.loc[:, 'Tweet'].apply(lambda x: TextBlob(x).sentiment[0]))


subjectivity = pd.Series(df_final.loc[:, 'Tweet'].apply(lambda x: TextBlob(x).sentiment[1]))



df_final = pd.concat([df_final, polarity, subjectivity], axis = 1)





def avg_word(sentence):
    words = sentence.split()
    if len(words) != 0:
        return (sum(len(word) for word in words)/len(words))
    else:
        return 0


df_final['avg_word'] = df_final.iloc[:, 0].apply(lambda x : avg_word(x))



df_final['stopwords'] = df_final.iloc[:, 0].apply(lambda x: len([x for x in x.split() if x in stop]))



X_train = df_final.iloc[:, 2:]



y_train = df_disease.iloc[:, 1]





X_train = ss.fit_transform(X_train)

X_train = pca.fit_transform(X_train)




knc = KNeighborsClassifier(n_neighbors = 7)
knc.fit(X_train, y_train)



df1_test = pd.read_csv('malaria_day_3_test.csv', header = 0, index_col = 0)



df2_test = pd.read_csv('dengue_day_3_test.csv', header = 0, index_col = 0)



df3_test = pd.read_csv('typhoid_day_3_test.csv', header = 0, index_col = 0)



df4_test = pd.read_csv('cancer_day_3_test.csv', header = 0, index_col = 0)



df5_test = pd.read_csv('tb_day_3_test.csv', header = 0, index_col = 0)



df_test = pd.DataFrame(columns = ['Tweet', 'Label'])


df1_test['Label'] = 0
df2_test['Label'] = 1
df3_test['Label'] = 2
df4_test['Label'] = 3
df5_test['Label'] = 4


df_test = pd.concat([df1_test.iloc[:, [1, 5]], df2_test.iloc[:, [1, 5]], df3_test.iloc[:, [1, 5]], df4_test.iloc[:, [1, 5]], df5_test.iloc[:, [1, 5]]], axis = 0)


df_test = df_test.sample(frac = 1).reset_index(drop = True)


for i in range(df_test.shape[0]):
    df_test.iloc[i, 0] = df_test.iloc[i, 0].lower()
      


for i in range(df_test.shape[0]):
    df_test.iloc[i, 0] = remove_punc(df_test.iloc[i, 0])

    
for i in range(df_test.shape[0]):
    df_test.iloc[i, 0] = clean_tweet(df_test.iloc[i, 0])



tfidf_matrix_test = tfidf.fit_transform(df_test.iloc[:, 0])



wordvec_df_test = tfidf_matrix_test.toarray()

wordvec_df_test = pd.DataFrame(wordvec_df_test)

df_final_test = pd.concat([df_test, wordvec_df_test], axis = 1)





polarity = pd.Series(df_final_test.loc[:, 'Tweet'].apply(lambda x: TextBlob(x).sentiment[0]))


subjectivity = pd.Series(df_final_test.loc[:, 'Tweet'].apply(lambda x: TextBlob(x).sentiment[1]))



df_final_test = pd.concat([df_final_test, polarity, subjectivity], axis = 1)





df_final_test['avg_word'] = df_final_test.iloc[:, 0].apply(lambda x : avg_word(x))




df_final_test['stopwords'] = df_final_test.iloc[:, 0].apply(lambda x: len([x for x in x.split() if x in stop]))





X_test = df_final_test.iloc[:, 2:]



y_test = df_final_test.iloc[:, 1]





#loaded_model.predict_proba(X_test)[15]


X_test = ss.fit_transform(X_test)

X_test = pca.fit_transform(X_test)





#loaded_model = joblib.load('knc_model.sav')


f1(y_test, knc.predict(X_test), average = 'micro')





ber = GaussianNB()

ber.fit(X_train, y_train)

f1(y_test, ber.predict(X_test), average = 'micro')



lr =LogisticRegression()

lr.fit(X_train, y_train)

lr.score(X_test, y_test)



rfc = RandomForestClassifier(n_estimators = 100, criterion = 'gini', max_features = 3, random_state = 1)

rfc.fit(X_train, y_train)

rfc.score(X_test, y_test)



dtc = DecisionTreeClassifier(criterion = 'gini', max_features = 3)

dtc.fit(X_train, y_train)

dtc.score(X_test, y_test)


gbc = GradientBoostingClassifier(max_features = 3)

gbc.fit(X_train, y_train)

gbc.score(X_test, y_test)


xgb = xgboost.XGBClassifier()

xgb.fit(X_train, y_train)

xgb.score(X_test, y_test)



km = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 1, verbose = 0, random_state = 3425)
km.fit(X_train)
labels = km.labels_

sil(X_train, labels, random_state = 1)

pred = km.predict(X_test)








filename = 'rfc_model.sav'
joblib.dump(rfc, filename)







