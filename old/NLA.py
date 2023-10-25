# -*- coding: utf-8 -*-
 
# ------------------------------------------------------------------------------------------------------------------------------------------- 
# 1 - IMPORTO EL DATAFRAME
# ------------------------------------------------------------------------------------------------------------------------------------------- 
# A - Puedo importar directamente el texto preprocesado y exportado en el paso 2 B al corpus. Saltar el paso 3.
file_path = "C:\\Users\\Fede\\Desktop\\GitHub\\hppd\\hppd_nla\\scraps\\preprocessed\\hppd_forum_all_preprocessed.txt"
file_dict = open(file_path, "r")
file_text = file_dict.readlines()
file_dict.close()        

corpus_docs = []
for doc in file_text:
    if doc != "\n":
        corpus_docs.append(doc)

del doc
del file_path
del file_dict
del file_text

# ----------------------------------------
# B - Puedo importar el XML scrapeado:
import pandas as pd
import xml.etree.ElementTree as ET
 
tree = ET.parse('C:\\Users\\Fede C\\Desktop\\GitHub\\hppd\\hppd_nla\\scraps\\hppd_forum_general.xml')

posts = []
for node in tree.getroot():        
    text = node[0].text      
    if text:  	    
        posts.append(text)
    
cols = ['text']    
dataframe = pd.DataFrame(posts, columns = cols)

# Guardo el dataframe original en un corpus por si quiero hacer algún análisis en crudo (ie, Sentiment Analysis)
crude_corpus_docs = []
for i in range(0, len(dataframe)):   
    crude_corpus_docs.append(dataframe['text'][i])   
    
# ----------------------------------------   
# D - Puedo importar texto plano de un archivo al dataframe (para hacer LDA/LSA con un texto cualquiera)
import pandas as pd

file_path = "C:\\Users\\Fede C\\Desktop\\GitHub\\hppd\\hppd_nla\\text_input.txt"
file_dict = open(file_path, "r")
file_text = file_dict.readlines()
file_dict.close()
cols = ['text']    
dataframe = pd.DataFrame(file_text, columns = cols)

# ------------------------------------------------------------------------------------------------------------------------------------------- 
# 2 - PREPROCESADO / CORPUS
# ------------------------------------------------------------------------------------------------------------------------------------------- 
# A - Preproceso el texto importado en 1
import re
import nltk
from nltk.corpus import stopwords  
from nltk.stem import WordNetLemmatizer

# descargo de nltk palabras que no sean relevantes para la predicción del texto (artículos, preposiciones, etc) y para lematizar:
nltk.download('stopwords') 
nltk.download('wordnet')

lmt = WordNetLemmatizer()
corpus_docs = []
for i in range(0, len(dataframe)):    
    # no quiero que remueva las letras de la A a la Z, ni en minúscula ni en mayúscula. también quiero que transforme lo que borra en un espacio 
    # y que luego paso todo a lower case:
    text = re.sub('[^a-zA-Z]', ' ', dataframe['text'][i])
    text = text.lower()
    # reformulo la variable string en una lista de elementos que contienen cada palabra del texto (split) y luego las chequeo contra stopwords 
    # en la iteración, descartando así las que no van:
    text = text.split()    
    text = [lmt.lemmatize(word) for word in text if not word in set(stopwords.words('english'))] 
    text = [word for word in text if len(word) >= 3]
    # vuelvo a transformar la lista de elementos en una concatenación de elementos separados por un espacio:
    text = ' '.join(text)    
    if text:
        corpus_docs.append(text)    

# ----------------------------------------
# B - Puedo exportar el texto preprocesado para la próxima vez cargar con 1 B
file_path = "C:\\Users\\Fede\\Desktop\\GitHub\\hppd\\hppd_nla\\scraps\\preprocessed\\hppd_forum_all_preprocessed.txt"
file_prepro = open(file_path, 'w+')
with file_prepro as f:
    for item in corpus_docs:
        if item != "\n":
            f.write("%s\n" % item)
file_prepro.close()

# ------------------------------------------------------------------------------------------------------------------------------------------- 
# 3 - REEMPLAZO DE PALABRAS EN DOCUMENTOS DEL CORPUS (OPCIONAL)
# ------------------------------------------------------------------------------------------------------------------------------------------- 
# A - Reemplazo de palabras específicas del preprocesado del corpus  
import re

lst = ["shroom" , "shrooms"]
rx = re.compile(r'\b(?:{})\b'.format("|".join(lst)))
corpus_docs = [rx.sub('mushroom', string) for string in corpus_docs]    

lst = ["cbd", "weed"]
rx = re.compile(r'\b(?:{})\b'.format("|".join(lst)))
corpus_docs = [rx.sub('cannabis', string) for string in corpus_docs]    

lst = ["opioids"]
rx = re.compile(r'\b(?:{})\b'.format("|".join(lst)))
corpus_docs = [rx.sub('opioid', string) for string in corpus_docs]

lst = ["suicide"]
rx = re.compile(r'\b(?:{})\b'.format("|".join(lst)))
corpus_docs = [rx.sub('suicidal', string) for string in corpus_docs]

lst = ["bpd"]
rx = re.compile(r'\b(?:{})\b'.format("|".join(lst)))
corpus_docs = [rx.sub('bipolar', string) for string in corpus_docs]

lst = ["gad"]
rx = re.compile(r'\b(?:{})\b'.format("|".join(lst)))
corpus_docs = [rx.sub('generalized', string) for string in corpus_docs]

lst = ["ocd"]
rx = re.compile(r'\b(?:{})\b'.format("|".join(lst)))
corpus_docs = [rx.sub('obsessive', string) for string in corpus_docs]

lst = ["ptsd"]
rx = re.compile(r'\b(?:{})\b'.format("|".join(lst)))
corpus_docs = [rx.sub('posttraumatic', string) for string in corpus_docs]

lst = ["benzo" , "benzos"]
rx = re.compile(r'\b(?:{})\b'.format("|".join(lst)))
corpus_docs = [rx.sub('benzodiacepine', string) for string in corpus_docs]   
# ----------------------------------------
lst = ["atd" , "imao", "ssri"]
rx = re.compile(r'\b(?:{})\b'.format("|".join(lst)))
corpus_docs = [rx.sub('antidepressant', string) for string in corpus_docs]

lst = ["atp"]
rx = re.compile(r'\b(?:{})\b'.format("|".join(lst)))
corpus_docs = [rx.sub('antipsychotic', string) for string in corpus_docs]

lst = ["desipramine", "amitriptyline", "nortriptyline", "clomipramine", "trazodone", "nefazodone", "fluoxetine", "bupropion", "sertraline", "paroxetine", "venlafaxine", "desvenlafaxine", "fluvoxamine", "mirtazapine", "citalopram", "escitalopram", "duloxetine", "vilazodone", "atomoxetine", "vortioxetine", "levomilnacipran", "phenelzine", "tranylcypromine", "selegiline", "norpramin", "elavil", "aventyl", "pamelor", "anafranil", "oleptro", "prozac", "sarafem", "wellbutrin", "zoloft", "paxil", "effexor", "pristiq", "luvox", "remeron", "celexa", "lexapro", "cymbalta", "viibryd", "strattera", "trintellix", "fetzima", "nardil", "parnate", "emsam"]
rx = re.compile(r'\b(?:{})\b'.format("|".join(lst)))
corpus_docs = [rx.sub('antidepressant', string) for string in corpus_docs] 

lst = ["diazepam", "chlordiazepoxide", "clonazepa", "lorazepam", "alprazolam", "buspirone", "gabapentin", "hydroxyzine", "propranolol", "atenolol", "guanfacine", "clonidine", "pregabalin", "prazosin", "valium", "librium", "klonopin", "ativan", "xanax", "buspar", "neurontin", "atarax", "vistaril", "inderal", "tenormin", "tenex", "intuniv", "catapres", "kapvay", "lyrica", "minipress"]
rx = re.compile(r'\b(?:{})\b'.format("|".join(lst)))
corpus_docs = [rx.sub('benzodiacepine', string) for string in corpus_docs] 

lst = ["temazepam", "triazolam", "zolpidem", "zaleplon", "eszopiclone", "ramelteon", "diphenhydramine", "doxepin", "suvorexant", "restoril", "halcion", "ambien", "intermezzo", "sonata", "lunesta", "rozerem", "benadryl", "silenor", "belsomra"]
rx = re.compile(r'\b(?:{})\b'.format("|".join(lst)))
corpus_docs = [rx.sub('hipnotic', string) for string in corpus_docs] 

lst = ["lithium", "carbonate", "carbamazepine", "divalproex", "lamotrigine", "oxcarbazepine", "eskalith", "lithonate", "symbyax", "tegretol", "equetro", "depakote", "lamictal", "trileptal"]
rx = re.compile(r'\b(?:{})\b'.format("|".join(lst)))
corpus_docs = [rx.sub('stabilizer', string) for string in corpus_docs] 

lst = ["chlorpromazine", "clozapine", "quetiapine", "perphenazine", "haloperidol", "pimozide", "risperidone", "paliperidone", "olanzapine", "ziprasidone", "iloperidone", "asenapine", "lurasidone", "aripiprazole", "brexpiprazole", "cariprazine", "thorazine", "clozari", "seroquel", "trilafon", "haldol", "orap", "risperdal", "invega", "zyprexa", "geodon", "fanapt", "saphris", "latuda", "abilify", "rexulti", "vraylar"]
rx = re.compile(r'\b(?:{})\b'.format("|".join(lst)))
corpus_docs = [rx.sub('antipsychotic', string) for string in corpus_docs] 

lst = ["methylphenidate", "dexmethylphenidate", "dextroamphetamine", "lisdexamphetamine", "damphetamine", "lamphetamine", "modafinil", "armodafanil", "salts", "sulfate", "ritalin", "concerta", "methylin", "daytrana", "quillivant", "focalin", "dexedrine", "vyvanse", "adderall", "provigil", "sparlon", "nuvigil", "mydayis", "adzenys", "evekeo"]
rx = re.compile(r'\b(?:{})\b'.format("|".join(lst)))
corpus_docs = [rx.sub('psychoestimulant', string) for string in corpus_docs] 

del rx
del lst

# ------------------------------------------------------------------------------------------------------------------------------------------- 
# 4 - FILTRO DE DOCUMENTOS POR PALABRAS (OPCIONAL)
# ------------------------------------------------------------------------------------------------------------------------------------------- 
# Armo el corpus sólo con los documentos que contengan ciertas palabras.
import re

# obtengo el diccionario de términos que necesite
file_path = "C:\\Users\\Fede C\\Desktop\\GitHub\\hppd\\hppd_nla\\dictionaries\\medtypes_list.txt"
file_dict = open(file_path, "r")
dict_list = file_dict.readlines()
file_dict.close()

terms = []
for i in range(0, len(dict_list)):    
    text = re.sub('[^a-zA-Z]', ' ', dict_list[i])
    text = text.strip()
    text = text.lower()
    text = text.split()            
    text = ' '.join(text)    
    if text:        
        terms.append(text)

#terms = ['disorder', 'syndrome']
# ----------------------------------------
        
# filtro los documentos
temp_corpus = []
for doc in corpus_docs:
    # puedo usar any() o puedo usar all() si es necesario.
    if any(term in doc for term in terms):  
        temp_corpus.append(doc)           
        
corpus_docs = []
corpus_docs = temp_corpus     

del doc
del temp_corpus
del file_path
del file_dict
del dict_list
del i 
del text

# ------------------------------------------------------------------------------------------------------------------------------------------- 
# 5 - DICCIONARIOS (OPCIONAL PRE PASO 4)
# -------------------------------------------------------------------------------------------------------------------------------------------      
import re

# obtengo el diccionario que necesite
file_path = "C:\\Users\\Fede\\Desktop\\GitHub\\hppd\\hppd_nla\\dictionaries\\meddrugs_list.txt"
file_dict = open(file_path, "r")
dict_list = file_dict.readlines()
file_dict.close()

# preproceso el texto del diccionario sin lematizar
dictionary = []
for i in range(0, len(dict_list)):    
    text = re.sub('[^a-zA-Z]', ' ', dict_list[i])
    text = text.strip()
    text = text.lower()
    text = text.split()            
    text = ' '.join(text)    
    if text:        
        dictionary.append(text)
        
del file_path
del file_dict
del dict_list
del i 
del text

# ------------------------------------------------------------------------------------------------------------------------------------------- 
# 6 - ARMO EL BoW 
# ------------------------------------------------------------------------------------------------------------------------------------------- 
# hago la tokenización eligiendo cuántas palabras voy a tomar de los documentos y con qué criterios
# max_df: ignora items que aparecen muy frecuentemente. 
#   max_df = 0.50 significa "ignorá los que aparecen en más del 50% de los documentos". Default = 1.0
#   max_df = 25 significa "ignorá los que aparecen en más de 25 documentos". 
# min_df: ignora items que aparecen muy infrecuentemente (palabras mal escritas, interjecciones, etc).
#   min_df = 0.01 significa "ignorá los que aparecen en menos del 1% de los documentos".
#   min_df = 5 significa "ignorá los que aparecen en menos de 5 documentos". Default = 1
# max_features: la cantidad máxima de palabras a utilizar. Corta por frecuencia (sobre las ya filtradas si hay max_df y min_df). 3000 palabras cubren el 95% de idioma inglés escrito.
# vocabulary: sólo se vectorizan las palabras que están en el diccionario, sin importar los parámetros _df.
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer  

count_vectorizer = CountVectorizer(min_df=0.05, max_df=0.95) 

# armo la sparse matrix
BoW = count_vectorizer.fit_transform(corpus_docs).toarray()

# armo el TFIDF
tfidfconverter = TfidfTransformer()  
BoW_tfidf = tfidfconverter.fit_transform(BoW).toarray() 

# evalúo con qué palabras se quedó
corpus_words = count_vectorizer.vocabulary_
corpus_words_sum = BoW.sum(axis=0)
        
# armo diccionario de término/frecuencia
wc_dict = {}
index = 0
for word in corpus_words:
    wc_dict[word] = corpus_words_sum[index]    
    index += 1

del index
del word    
del corpus_words_sum
del corpus_words            

# ------------------------------------------------------------------------------------------------------------------------------------------- 
# 7 - WORDCLOUD
# ------------------------------------------------------------------------------------------------------------------------------------------- 
# Puedo visualizar el corpus generado con los parámetros del count_vectorizer en el paso 4.
from wordcloud import WordCloud
import matplotlib.pyplot as plt 
    
wordcloud = WordCloud(width = 1000, height = 1000, 
                background_color ='white', 
                min_font_size = 10).generate_from_frequencies(wc_dict)
                     
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0)
plt.show()

del wordcloud


# ------------------------------------------------------------------------------------------------------------------------------------------- 
# 8 A - LSA GENSIM
# ------------------------------------------------------------------------------------------------------------------------------------------- 
import nltk
from gensim import models, corpora
from gensim.models import LsiModel
from gensim.models.coherencemodel import CoherenceModel
from nltk import word_tokenize
import matplotlib.pyplot as plt

nltk.download('punkt')

NUM_TOPICS = 10
NUM_WORDS = 6

# Tokenizo los documentos importados en el paso 1 
tokenized_corpus = []
for text in corpus_docs:
    tokenized_corpus.append(word_tokenize(text))
  
# Armo un diccionario asociando cada palabra a un word id
lsa_dictionary = corpora.Dictionary(tokenized_corpus)
 
# Asocio el word id a cada palabra para cada documento
lsa_doc_terms = [lsa_dictionary.doc2bow(text) for text in tokenized_corpus]

# ----------------------------------------
# Computo Coherence y determino la cantidad de topics adecuada
stop = NUM_TOPICS 
start = 2
step = 1

coherence_values = []
model_list = []
for num_topic in range(start, stop, step):
    # generate LSA model
    # print(str(num_topics), " of ", str(NUM_TOPICS))
    lsa_model_gsm = LsiModel(lsa_doc_terms, num_topics = num_topic, id2word = lsa_dictionary)
    model_list.append(lsa_model_gsm)
    coherencemodel = CoherenceModel(model = lsa_model_gsm, texts = tokenized_corpus, dictionary = lsa_dictionary, coherence='c_v')
    coherence_values.append(coherencemodel.get_coherence())

# ----------------------------------------
# Ploteo los topics
x = range(start, stop, step)
plt.plot(x, coherence_values)
plt.xlabel("Number of Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

del start, step, stop
del text
del tokenized_corpus
del num_topic

# ----------------------------------------
# Muestro los topics para la cantidad de modelos óptima:
TOPIC_COUNT = 5
lsa_model_gsm = LsiModel(lsa_doc_terms, num_topics = TOPIC_COUNT, id2word = lsa_dictionary)

for idx in range(TOPIC_COUNT):
    # muestro las palabras que mas contribuyen a cada topic    
    print("Topic #%s:" % idx, lsa_model_gsm.print_topic(idx, NUM_WORDS))          
        
del idx

# ------------------------------------------------------------------------------------------------------------------------------------------- 
# 8 B - LDA SKLEARN
# ------------------------------------------------------------------------------------------------------------------------------------------- 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

NUM_TOPICS = 10
NUM_WORDS = 10
    
count_vectorizer = CountVectorizer(max_df=0.95, min_df=0.05, max_features=1000)
# armo la sparse matrix 
BoW = count_vectorizer.fit_transform(corpus_docs)
# evalúo con qué palabras se quedó
features = count_vectorizer.get_feature_names()
lda_model_skl = LatentDirichletAllocation(n_topics=NUM_TOPICS, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(BoW)

# Muestro los topics para el corpus dado:
for idx, topic in enumerate(lda_model_skl.components_):
    print("Topic %d:" % (idx))
    print(" ".join([features[i]
                    for i in topic.argsort()[:-NUM_WORDS - 1:-1]]))
    
del idx
del features

# ------------------------------------------------------------------------------------------------------------------------------------------- 
# 8 C - LDA GENSIM
# ------------------------------------------------------------------------------------------------------------------------------------------- 
import nltk
from gensim import corpora
from nltk import word_tokenize
nltk.download('punkt')

NUM_TOPICS = 10
NUM_WORDS = 10

# Tokenizo los documentos importados en el paso 1 
tokenized_corpus = []
for text in corpus_docs:
    tokenized_corpus.append(word_tokenize(text))
  
# Armo un diccionario asociando cada palabra a un word id
lda_dictionary = corpora.Dictionary(tokenized_corpus)
 
# Asocio el word id a cada palabra para cada documento
lda_doc_terms = [lda_dictionary.doc2bow(text) for text in tokenized_corpus]
 
# Armo el modelo LDA
lda_model_gsm = models.LdaModel(corpus = lda_doc_terms, alpha = 'auto', num_topics = NUM_TOPICS, id2word = lda_dictionary)

# Muestro los topics para el corpus dado:
for idx in range(NUM_TOPICS):
    # muestro las palabras que mas contribuyen a cada topic    
    print("Topic #%s:" % idx, lda_model_gsm.print_topic(idx, NUM_WORDS))          

# Evalúo qué temas contiene el corpus y en qué proporción
#bow = lda_dictionary.doc2bow(tokenized_corpus[64])
#print('\nWeights: ', lda_model_gsm[bow])

# ----------------------------------------
# Computo Coherence para Gensim
from gensim.models import CoherenceModel

coherence_model_lda = CoherenceModel(model=lda_model_skl, texts=tokenized_corpus, dictionary=lda_dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence: ', coherence_lda)

del coherence_model_lda

# ------------------------------------------------------------------------------------------------------------------------------------------- 
# 9 - pyLDAvis para LDA
# ------------------------------------------------------------------------------------------------------------------------------------------- 
import pyLDAvis.gensim, pyLDAvis.sklearn
import os, webbrowser

file_path = 'C:\\Users\\Fede C\\Desktop\\GitHub\\hppd\\hppd_nla\\'

pyLDAvis.enable_notebook()

# Para Gensim:
#vis = pyLDAvis.gensim.prepare(lda_model_gsm, lda_corpus, dictionary=lda_dictionary)
#file_name = "LDAvis_gsm.html"

# Para Sklearn:
vis = pyLDAvis.sklearn.prepare(lda_model_skl, BoW, count_vectorizer)
file_name = "LDAvis_skl.html"

pyLDAvis.save_html(vis, file_path + file_name)

webbrowser.open('file://' + os.path.realpath(file_path + file_name))

# ------------------------------------------------------------------------------------------------------------------------------------------- 
# 10 - SENTIMENT ANALYSIS
# ------------------------------------------------------------------------------------------------------------------------------------------- 
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()
corpus_sa_scores = np.zeros((len(corpus_docs), 4))
i=0
for phrase in corpus_docs:            
    sa_score = analyser.polarity_scores(phrase)    
    #print("{:-<40} {}".format(sentence, str(score)))
    corpus_sa_scores[i,0] = sa_score['neg']
    corpus_sa_scores[i,1] = sa_score['neu']
    corpus_sa_scores[i,2] = sa_score['pos']
    corpus_sa_scores[i,3] = sa_score['compound']
    i += 1   
    
print(corpus_sa_scores.mean(0)) 

del phrase
del sa_score
del i

# ------------------------------------------------------------------------------------------------------------------------------------------- 
# 11 - MATRICES DE CORRELACIÓN
# ------------------------------------------------------------------------------------------------------------------------------------------- 
import numpy as np
# Genero una matriz de co ocurrencia entre las palabras.

word_corr = np.zeros(shape=(len(dictionary), len(dictionary)))
word_corr_frec = np.zeros(shape=(len(dictionary), len(dictionary)))
word_corr_frec_cond = np.zeros(shape=(len(dictionary), len(dictionary)))

BoW_mask = BoW > 0
# matriz de co ocurrencia booleana 
word_corr = np.dot(BoW_mask.T, BoW_mask)
# matriz de co ocurrencia (utilizo la máscara y si una palabra aparece más de una vez por documento se computa como una sola)
word_corr_frec = np.dot(BoW_mask.T.astype(float), BoW_mask.astype(float))
# matriz de co ocurrencia completa y condicional (si una palabra aparece más de una vez por documento se computan todas las veces)
# Cuando en un documento figura la palabra [columna] aparece la palabra [línea] [valor] veces:
#       word1  word2  word3         #       word1  word2  word3
# doc0    3      4      2           # word1  17      9     11
# doc1    6      1      0           # word2   5      5      4
# doc3    8      0      4           # word3   6      2      6
word_corr_frec_cond = np.dot(BoW.T, BoW_mask) 

del BoW_mask

# ----------------------------------------
# Normalizo una matriz.
matrix = word_corr_frec_cond
corr_normalized = np.zeros(shape=(len(dictionary), len(dictionary)))
rows = len(matrix)    
cols = len(matrix[0]) 
for i in range(rows):
    count = int(wc_dict[dictionary[i]])    
    for j in range(cols):                
        corr_normalized[i][j] = float(matrix[i][j] / float(count))

del count
del rows, cols
del j, i
del matrix

# ------------------------------------------------------------------------------------------------------------------------------------------- 

