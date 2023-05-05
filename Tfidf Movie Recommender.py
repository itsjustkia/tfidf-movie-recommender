#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


import json
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


# In[3]:


df = pd.read_csv("tmdb_5000_movies.csv")


# In[4]:


df.head()


# In[9]:


x=df.iloc[0]
x
x['genres']
x['keywords']


# In[13]:


def genres_and_keywords_to_string(row):
    genres = json.loads(row['genres'])
    genres = ' '.join(''.join(j['name'].split()) for j in genres)
    
    keywords = json.loads(row['keywords'])
    keywords = ' '.join(''.join(j['name'].split()) for j in keywords)
    return "%s %s" % (genres, keywords)


# In[14]:


df['string'] = df.apply(genres_and_keywords_to_string, axis=1)


# In[15]:


df['string']


# In[16]:


tfidf = TfidfVectorizer(max_features=2000)


# In[17]:


X = tfidf.fit_transform(df['string'])


# In[19]:


X


# In[20]:


movie2index = pd.Series(df.index, index=df['title'])


# In[21]:


def recommend(title):
    idx = movie2index[title]
    query=X[idx]
    scores = cosine_similarity(query, X)
    scores = scores.flatten()
    recommend_idx = (-scores).argsort()[1:6]
    return df['title'].iloc[recommend_idx]


# In[22]:


print(recommend('Scream 3'))


# In[23]:


print(recommend('The Dark Knight Rises'))


# In[24]:


print(recommend("Pirates of the Caribbean: At World's End"))


# In[ ]:




