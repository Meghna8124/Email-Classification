#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from collections import Counter


# In[2]:


def get_words_labels(directory):
    words = []
    labels = []
    for filename in os.listdir(os.getcwd() + '/lingspamPublic/bare/' + directory):
        with open (os.getcwd() + '/lingspamPublic/bare/' + directory + '/' + filename) as fin:
            email_words = word_tokenize(fin.read())
            if filename.startswith('spm'):
                label = 1
            else:
                label = 0
            labels.append(label)
            words = words + email_words
            
    
    return words, labels


# In[3]:


def get_dictionaries(word_list):
    word_list = sorted(list(set(word_list)))
    word_list = [item for item in word_list if item.isalpha()]
    num_list = list(range(len(word_list)))
    word_list.remove('Subject')
    
    dict1 = dict(zip(word_list, num_list))
    
    stop_words = set(stopwords.words('english'))
    word_list_sw = [w for w in word_list if not w in stop_words]
    num_list = list(range(len(word_list_sw)))
    
    dict2 = dict(zip(word_list_sw, num_list))
    
    lemmatizer = WordNetLemmatizer()
    word_list_lem = [lemmatizer.lemmatize(w) for w in word_list_sw]
    word_list_lem = sorted(list(set(word_list_lem)))
    num_list = list(range(len(word_list_lem)))
    
    dict3 = dict(zip(word_list_lem, num_list))
    
    dict_freq = Counter(word_list)
    for key in list(dict_freq):
        if dict_freq[key] <6000 and dict_freq[key]>20:
            del dict_freq[key]
    pruned_list = list(dict_freq.keys())
    word_list_pruned = [w for w in word_list_lem if w not in pruned_list]
    num_list = list(range(len(word_list_pruned)))
    dict4 = dict(zip(word_list_pruned, num_list))
    
    return dict1, dict2, dict3, word_list_lem


# In[4]:


def get_array(dir_list, dictionary):
    
    clf = CountVectorizer(input = 'filename', vocabulary = dictionary)
   
    file_list = []
    for directory in dir_list:
        #for filename in os.listdir(os.getcwd() + '/lingspamPublic/bare/' + directory):
        path = os.getcwd() + '/lingspamPublic/bare/' + directory + '/'
        file_list = file_list + [os.path.join(path, x) for x in os.listdir(path)]
        #print(len(os.listdir(path)))

    X = clf.fit_transform(file_list)
    clf.get_feature_names()

    return X.toarray()


# In[5]:


list_dir = os.listdir(os.getcwd() + '/lingspamPublic/bare')
list_dir.remove('part10')
list_dir.append('part10')

words = []
labels = []
for d in list_dir:
    w_list, lbl = get_words_labels(d)
    
    words.append(w_list)
    labels.append(lbl)


# In[6]:


acc = np.zeros((10, 1))
for j in range(10):
    list1 = []
    y_train = []
    for i in range(10):
        if not i==j:
            list1 = list1 + words[i]
            y_train = y_train + labels[i]
    dict1, dict2, dict3, dict_list_lem = get_dictionaries(list1)
    temp_list = list_dir.copy()
    temp_list.remove('part' + str(j+1))
    
    dict_freq = Counter(list1)
    for key in list(dict_freq):
            if dict_freq[key] <6000 and dict_freq[key]>20:
                del dict_freq[key]
    pruned_list = list(dict_freq.keys())
    word_list_pruned = [w for w in dict_list_lem if w not in pruned_list]
    num_list = list(range(len(word_list_pruned)))
    dict4 = dict(zip(word_list_pruned, num_list))
    
    X = get_array(temp_list, dict4)
    y_train = np.array(y_train)
    model = MultinomialNB()
    model.fit(X, y_train)    
    X_test = get_array(['part' + str(j+1)], dict4)
    y_test = labels[j]
    y_test = np.array(y_test)
    p = model.predict(X_test)
    acc[j][0] = accuracy_score(y_test, p)


# In[7]:


print(acc)
print(np.average(acc))


# In[ ]:




