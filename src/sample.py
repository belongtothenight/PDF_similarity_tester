import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
import numpy as np

# Source: https://dev.to/thepylot/compare-documents-similarity-using-python-nlp-4odp

# original file
file_docs = []
with open('../similarity_test/original.txt') as f:
    tokens = sent_tokenize(f.read())
    for line in tokens:
        file_docs.append(line)
print("Number of documents:",len(file_docs))

gen_docs = [[w.lower() for w in word_tokenize(text)] for text in file_docs]
print(gen_docs)

dictionary = gensim.corpora.Dictionary(gen_docs)
print(dictionary.token2id)

corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
print(corpus)

tf_idf = gensim.models.TfidfModel(corpus)
print(tf_idf)
for doc in tf_idf[corpus]:
    print([[dictionary[id], np.around(freq, decimals=2)] for id, freq in doc])

sims = gensim.similarities.Similarity('./cache/',tf_idf[corpus], num_features=len(dictionary))
print(sims)

# file to compare with
file2_docs = []
with open('../similarity_test/comparing.txt') as f:
    tokens = sent_tokenize(f.read())
    for line in tokens:
        file2_docs.append(line)
print("Number of documents:",len(file2_docs))
for line in file2_docs:
    query_doc = [w.lower() for w in word_tokenize(line)]
    query_doc_bow = dictionary.doc2bow(query_doc)
    print(query_doc_bow)
    query_doc_tf_idf = tf_idf[query_doc_bow]
    print(query_doc_tf_idf)
    print(sims[query_doc_tf_idf])