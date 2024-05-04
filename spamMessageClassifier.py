#!/usr/bin/env python
# coding: utf-8

# # Spam message classifier

# In[236]:


import numpy as np
import pandas as pd


# In[237]:


df = pd.read_csv('spam.csv', encoding='latin1')


# In[238]:


df.sample(5)


# In[239]:


df.shape


# # 1.Data Cleaning

# In[240]:


df.info()


# In[241]:


# as 3 cloumns have very less non-null values,means tere are missing values so we are dropping that 3v columns 
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)


# In[242]:


df.sample(5)


# In[243]:


df.rename(columns={'v1':'target','v2':'text'},inplace=True)   #using dictionary
df.sample(5)


# In[244]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder() #label incoder spam or ham ko 1 and 0 mai convert krne k liye


# In[245]:


df['target'] = encoder.fit_transform(df['target'])


# In[246]:


df.head()


# In[247]:


#checking missing values
df.isnull().sum()


# In[248]:


# checking duplicate values
df.duplicated().sum()


# In[249]:


# remove duplicates
df = df.drop_duplicates(keep='first') #first' means to keep the first occurrence of each duplicate row and remove the subsequent duplicates.


# In[250]:


df.duplicated().sum()


# In[251]:


df.shape


# # 2.Exploratory Data Analysis

# In[252]:


df.head()


# In[253]:


df['target'].value_counts() #kitne spam hai kitne non-spam hai


# In[254]:


import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(), labels=['ham','spam'],autopct="%0.2f") #0.2 mtlab . k baad 2 values
plt.show()


# In[255]:


# Data is imbalanced


# In[256]:


import nltk       #natural language toolKit Obviously


# In[257]:


nltk.download('punkt')     #dependencies


# In[258]:


df['num_characters'] = df['text'].apply(len)     # 1 sentence mai kitne words hai vo bata dega


# In[259]:


df.head()


# In[260]:


# num of words
df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))
# nltk.word_tokenize(x) jo hai 1 sentence ko words mai array mai breakdown kr dega
# or fir hum unn words kin length pta kr lenge using len
#lambda x:   This defines an anonymous function that takes a single argument, denoted as x.


# In[261]:


df.head()


# In[262]:


df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))
#sentence k bases p array and fir len se sentence ki length mtlb kitne length hai


# In[263]:


df.head()


# In[264]:


df[['num_characters','num_words','num_sentences']].describe()


# In[265]:


# ham
df[df['target'] == 0][['num_characters','num_words','num_sentences']].describe()


# In[266]:


#spam
df[df['target'] == 1][['num_characters','num_words','num_sentences']].describe()


# In[267]:


import seaborn as sns


# In[268]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_characters'])
sns.histplot(df[df['target'] == 1]['num_characters'],color='red')


# In[269]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_words'])
sns.histplot(df[df['target'] == 1]['num_words'],color='red')


# In[270]:


sns.pairplot(df,hue='target')
#pairplot is a type of plot that shows pairwise relationships between variables in a dataset.
#hue='target', you're instructing Seaborn to use the values in the 'target' column of your DataFrame to determine the colors of the data points in the pairplot


# In[271]:


sns.heatmap(df.corr(),annot=True)   #correlation


# # 3.Data preprocessing
# 

# #### Lower case, Tokenization(words mai diviede krna), Removing special characters, Removing stop words(a, is,the,an,are basicallly vo words jinka sentence k matlab se koi koi role nai hota), punctuation and Stemming(mtlb dancing,danced,dance itno vo dance mai convert kr dega)

# In[272]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string  #for puncuation


nltk.download('stopwords')
nltk.download('punkt')

ps = PorterStemmer()

print(stopwords.words('english'))
print(string.punctuation)


# In[273]:


def transform_text(text):
    text = text.lower()     #lowercase
    text = word_tokenize(text)   #words mai divedie kiya
    
    y = []
    for i in text:
        if i.isalnum():   #special charecter htane k liye
            y.append(i)
    
    text = y[:]      #list directly copy nai hoti, clonning krna hota hai
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:    #stopwords and puncutuation hta diya
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))   #stemming kr di, kisi bhi word ko uske root word mai le aata hai
    
            
    return " ".join(y)   #string bana k return


# In[274]:


transformed_text = transform_text("I'm gonna be home soon hello ok and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.")
print(transformed_text)


# In[275]:


df['transformed_text'] = df['text'].apply(transform_text)  


# In[276]:


df.head()


# In[278]:


get_ipython().system('pip install wordcloud       #jo sbse important ya jyada use hone vale words hai usko bada krke dikhaega')


# In[279]:


from wordcloud import WordCloud
wc = WordCloud(width=500,height=500,min_font_size=10,background_color='black')#object bna diya, UI k sath


# In[280]:


spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep=" "))    #i yani spam vale messages k liye and string ko concatinate kr denge using space


# In[281]:


plt.figure(figsize=(15,6)) #size define kri
plt.imshow(spam_wc)


# In[282]:


ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))


# In[283]:


plt.figure(figsize=(15,6))
plt.imshow(ham_wc)


# In[284]:


df.head()


# In[285]:


spam_corpus = []   #empty list
for msg in df[df['target'] == 1]['transformed_text'].tolist():  #list of string/item
    for word in msg.split():     #This splits the message into individual words based on whitespace.
        spam_corpus.append(word)   #Each word extracted from the split message is appended to the spam_corpus list


# In[286]:


print(spam_corpus)
len(spam_corpus)


# In[287]:


from collections import Counter

word_counter = Counter(spam_corpus)  #1 dictionary create kr dega jha p humare corpus mai harr word kitni baar aaya hai vo bataega. for eg-> 'free':191
#Counter class, which is a container that keeps track of the count of elements. It is commonly used for counting the occurrences of elements in iterable objects like lists

top_words_df = pd.DataFrame(word_counter.most_common(30), columns=['Word', 'Frequency']) # 1 dataframe bnaya wordCounter ka fir most_common(30) jo humkon30 most common woccuring words nikal k dega and then column k naam rkh diya


sns.barplot(data=top_words_df, x='Word', y='Frequency')
plt.xticks(rotation='vertical')  #words ka rotation hai ye
plt.show()


# In[288]:


ham_corpus = []
for msg in df[df['target'] == 0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)


# In[289]:


print(ham_corpus)
len(ham_corpus)


# In[290]:


from collections import Counter
word_counter = Counter(ham_corpus)


top_words_df = pd.DataFrame(word_counter.most_common(30), columns=['Word', 'Frequency'])

sns.barplot(data=top_words_df, x='Word', y='Frequency')
plt.xticks(rotation='vertical')
plt.show()


# In[291]:


df.head()


# # Model Building 
# 

# #### Naive Bayes give brilliant performance on textual data

# In[329]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)


# In[330]:


X = tfidf.fit_transform(df['transformed_text']).toarray()


# In[331]:


X.shape


# In[332]:


y = df['target'].values


# In[333]:


from sklearn.model_selection import train_test_split


# In[360]:


from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


# In[361]:


gnb = GaussianNB()
bnb = BernoulliNB()


# In[362]:


gnb.fit(X_train,y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[363]:


bnb.fit(X_train,y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


# In[364]:


# tfidf --> MNB


# In[365]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


# In[366]:


svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
bnb = BernoulliNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)
xgb = XGBClassifier(n_estimators=50,random_state=2)


# In[367]:


clfs = {
    'SVC' : svc,
    'KN' : knc, 
    'NB': bnb, 
    'DT': dtc, 
    'LR': lrc, 
    'RF': rfc, 
    'AdaBoost': abc, 
    'BgC': bc, 
    'ETC': etc,
    'GBDT':gbdt,
    'xgb':xgb
}


# In[368]:


def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    
    return accuracy,precision


# In[369]:


train_classifier(svc,X_train,y_train,X_test,y_test)


# In[370]:


accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    
    current_accuracy,current_precision = train_classifier(clf, X_train,y_train,X_test,y_test)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)


# In[371]:


performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)
performance_df


# In[372]:


performance_df1 = pd.melt(performance_df, id_vars = "Algorithm")
performance_df1


# In[373]:


sns.catplot(x = 'Algorithm', y='value', 
               hue = 'variable',data=performance_df1, kind='bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation='vertical')
plt.show()


# In[374]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_3000':accuracy_scores,'Precision_max_ft_3000':precision_scores}).sort_values('Precision_max_ft_3000',ascending=False)


# In[375]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_scaling':accuracy_scores,'Precision_scaling':precision_scores}).sort_values('Precision_scaling',ascending=False)


# In[376]:


new_df = performance_df.merge(temp_df,on='Algorithm')


# In[377]:


new_df_scaled = new_df.merge(temp_df,on='Algorithm')


# In[378]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_num_chars':accuracy_scores,'Precision_num_chars':precision_scores}).sort_values('Precision_num_chars',ascending=False)


# In[379]:


new_df_scaled.merge(temp_df,on='Algorithm')


# In[380]:


# Voting Classifier
svc = SVC(kernel='sigmoid', gamma=1.0,probability=True)
bnb = BernoulliNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)

from sklearn.ensemble import VotingClassifier


# In[381]:


voting = VotingClassifier(estimators=[('svm', svc), ('nb', bnb), ('et', etc)],voting='soft')


# In[382]:


voting.fit(X_train,y_train)


# In[383]:


y_pred = voting.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# In[384]:


# Applying stacking
estimators=[('svm', svc), ('nb', bnb), ('et', etc)]
final_estimator=RandomForestClassifier()


# In[385]:


from sklearn.ensemble import StackingClassifier


# In[386]:


clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)


# In[387]:


clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# In[388]:


import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))


# In[ ]:




