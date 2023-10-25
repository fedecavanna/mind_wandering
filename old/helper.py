# -*- coding: utf-8 -*-
"""
Created on Sat May 25 17:39:45 2019

@author: Fede
"""

# ------------------------------------------------------------------------------------------------------------------------------------------- 
# NMF CON TFIDF

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

NUM_TOPICS = 15
NUM_WORDS = 10

tfidf_vectorizer = TfidfVectorizer(max_features=1000)
tfidf_model = tfidf_vectorizer.fit_transform(corpus_docs)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

nmf = NMF(n_components=NUM_TOPICS, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf_model)

for topic_idx, topic in enumerate(nmf.components_):
    print("Topic %d:" % (topic_idx))
    print(" ".join([tfidf_feature_names[i]
                    for i in topic.argsort()[:-NUM_WORDS - 1:-1]]))

# ------------------------------------------------------------------------------------------------------------------------------------------- 
# Guardado y carga de diccionarios / listas en numpy
# np.save(file_path, dictionary) # guardo lista/diccionario

import numpy as np

filepath = "C:\\Users\\Fede C\\Desktop\\GitHub\\results\\drugs\\"
wordcount_dict = np.load(filepath + "wordcount.npy").item() 
word_list = np.load(filepath + "dictionary.npy") 
corr_frecuencia = np.load(filepath + "corr_frecuencia.npy") 
corr_frecuencia_condicional = np.load(filepath + "corr_frecuencia_condicional.npy") 
# ------------------------------------------------------------------------------------------------------------------------------------------- 

#exporto la matriz
filepath = "C:\\Users\\Fede\\Desktop\\GitHub\\results\\meds\\drugs\\"
np.savetxt(filepath + "meddrugs_norm.csv", corr_normalized, delimiter=",")
# exporto la lista
data = ','.join(dictionary)
fh = open(filepath + "items.csv", 'w+')
fh.write(data)
fh.close()

# corto a partir de cierta frecuencia de palabras
exp = []
for word in wc_dict:
    if wc_dict[word] >= 19:
        print(word + ' ' + str(wc_dict[word]))
        exp.append(word)
        
filepath = "c:\\Users\\Fede\\Desktop\\GitHub\\hppd\\hppd_nla\\dictionaries\\meddrugs_list.txt"
fh = open(filepath, 'w+')
with fh as f:
    for word in exp:        
        f.write("%s\n" % word)
fh.close()