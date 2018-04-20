## Vocabulary Embedding

This file contains commands written in orderly manner equivalent to corresponding jupyter notebook with ipython3.

Commands :


FN = 'vocabulary-embedding'  

seed=16  

vocab_size = 30000  

embedding_dim = 50  

lower = False # dont lower case the text  

pwd  

import json  
fndata = '_dataset/sample-1M.jsonl'  
heads = []  
desc = []  
counter = 0  
with open(fndata) as f:  
&emsp;&emsp;for line in f:  
&emsp;&emsp;&emsp;if counter < 50000 :  
&emsp;&emsp;&emsp;&emsp;jdata = json.loads(line)    # for json lines file, loading line by line  
&emsp;&emsp;&emsp;&emsp;heads.append(jdata["title"].lower())  
&emsp;&emsp;&emsp;&emsp;desc.append(jdata["content"].lower())  
&emsp;&emsp;&emsp;&emsp;counter +=1  

if lower:  
&emsp;heads = [h.lower() for h in heads]  

if lower:
&emsp;desc = [h.lower() for h in desc]

len(desc),len(set(desc))  

len(heads),len(set(heads))  

from collections import Counter  
from itertools import chain  
def get_vocab(lst):  
&emsp;vocabcount = Counter(w for txt in lst for w in txt.split())  
&emsp;vocab = map(lambda x: x[0], sorted(vocabcount.items(), key=lambda x: -x[1]))  
&emsp;return vocab, vocabcount  

vocab, vocabcount = get_vocab(heads+desc)  

import matplotlib.pyplot as plt  
plt.plot([vocabcount[w] for w in vocab]);  
plt.gca().set_xscale("log", nonposx='clip')  
plt.gca().set_yscale("log", nonposy='clip')  
plt.title('word distribution in headlines and discription')  
plt.xlabel('rank')  
plt.ylabel('total appearances');  

empty = 0 # RNN mask of no data  
eos = 1  # end of sentence  
start_idx = eos+1 # first real word  

def get_idx(vocab, vocabcount):  
&emsp;word2idx = dict((word, idx+start_idx) for idx,word in enumerate(vocab))  
&emsp;word2idx['<empty>'] = empty  
&emsp;word2idx['<eos>'] = eos  
&emsp;idx2word = dict((idx,word) for word,idx in word2idx.items())  
&emsp;# Important changes in python 3  
&emsp;# Removed dict.iteritems(), dict.iterkeys(), and dict.itervalues().  
&emsp;# Instead: use dict.items(), dict.keys(), and dict.values() respectively.  
&emsp;return word2idx, idx2word  

word2idx, idx2word = get_idx(vocab, vocabcount)  

from keras.utils.data_utils import get_file  

fname = 'glove.6B.%dd.txt'%embedding_dim  
import os  
datadir_base = os.path.expanduser(os.path.join('~', '.keras'))  
if not os.access(datadir_base, os.W_OK):  
    datadir_base = os.path.join('/tmp', '.keras')  
datadir = os.path.join(datadir_base, 'datasets')  
glove_name = os.path.join(datadir, fname)  

if not os.path.exists(glove_name):  
&emsp;path = 'glove.6B.zip'  
&emsp;path = get_file(path, origin="http://nlp.stanford.edu/data/glove.6B.zip")  
&emsp;!unzip {path}  

glove_name = '/home/ash1sh/headlines/glove.6B.50d.txt'  

glove_n_symbols = !wc -l {glove_name}  
glove_n_symbols = int(glove_n_symbols[0].split()[0])  
glove_n_symbols  

glove_index_dict = {}  
glove_embedding_weights = np.empty((glove_n_symbols, embedding_dim))  
globale_scale=.1  
with open(glove_name, 'r') as fp:  
&emsp;i = 0  
&emsp;for l in fp:  
&emsp;&emsp;l = l.strip().split()  
&emsp;&emsp;w = l[0]  
&emsp;&emsp;glove_index_dict[w] = i  
&emsp;&emsp;glove_embedding_weights[i,:] = list(map(float,l[1:]))  
&emsp;&emsp;i += 1  
glove_embedding_weights *= globale_scale  


for w,i in glove_index_dict.items():  
&emsp;w = w.lower()  
&emsp;if w not in glove_index_dict:  
&emsp;&emsp;glove_index_dict[w] = i  


import numpy as np  
# generate random embedding with same scale as glove  
np.random.seed(seed)  
shape = (vocab_size, embedding_dim)  
scale = glove_embedding_weights.std()*np.sqrt(12)/2 # uniform and not normal  
embedding = np.random.uniform(low=-scale, high=scale, size=shape)  
print ()'random-embedding/glove scale', scale, 'std', embedding.std())  
# copy from glove weights of words that appear in our short vocabulary (idx2word)  
c = 0  
for i in range(vocab_size):  
&emsp;w = idx2word[i]  
&emsp;g = glove_index_dict.get(w, glove_index_dict.get(w.lower()))  
&emsp;if g is None and w.startswith('#'): # glove has no hashtags (I think...)  
&emsp;&emsp;w = w[1:]  
&emsp;&emsp;g = glove_index_dict.get(w, glove_index_dict.get(w.lower()))  
&emsp;if g is not None:  
&emsp;&emsp;embedding[i,:] = glove_embedding_weights[g,:]  
&emsp;&emsp;c+=1  
print ('number of tokens, in small vocab, found in glove and copied to embedding', c,c/float(vocab_size))  

glove_thr = 0.5  

word2glove = {}  
for w in word2idx:  
&emsp;if w in glove_index_dict:  
&emsp;&emsp;g = w  
&emsp;elif w.lower() in glove_index_dict:  
&emsp;&emsp;g = w.lower()  
&emsp;elif w.startswith('#') and w[1:] in glove_index_dict:  
&emsp;&emsp;g = w[1:]  
&emsp;elif w.startswith('#') and w[1:].lower() in glove_index_dict:  
&emsp;&emsp;g = w[1:].lower()  
&emsp;else:  
&emsp;&emsp;continue  
&emsp;word2glove[w] = g  

normed_embedding = embedding/np.array([np.sqrt(np.dot(gweight,gweight)) for gweight in embedding])[:,None]

nb_unknown_words = 100

glove_match = []  
for w,idx in word2idx.items():  
&emsp;if idx >= vocab_size-nb_unknown_words and w.isalpha() and w in word2glove:  
&emsp;&emsp;gidx = glove_index_dict[word2glove[w]]  
&emsp;&emsp;gweight = glove_embedding_weights[gidx,:].copy()  
&emsp;&emsp;# find row in embedding that has the highest cos score with gweight  
&emsp;&emsp;gweight /= np.sqrt(np.dot(gweight,gweight))  
&emsp;&emsp;score = np.dot(normed_embedding[:vocab_size-nb_unknown_words], gweight)  
&emsp;&emsp;while True:  
&emsp;&emsp;&emsp;embedding_idx = score.argmax()  
&emsp;&emsp;&emsp;s = score[embedding_idx]  
&emsp;&emsp;&emsp;if s < glove_thr:  
&emsp;&emsp;&emsp;&emsp;break  
&emsp;&emsp;&emsp;if idx2word[embedding_idx] in word2glove :  
&emsp;&emsp;&emsp;&emsp;glove_match.append((w, embedding_idx, s))  
&emsp;&emsp;&emsp;&emsp;break  
&emsp;&emsp;&emsp;score[embedding_idx] = -1  
glove_match.sort(key = lambda x: -x[2])  
print ('# of glove substitutes found', len(glove_match))  

for orig, sub, score in glove_match[-10:]:  
&emps;print (score, orig,'=>', idx2word[sub])  

glove_idx2idx = dict((word2idx[w],embedding_idx) for  w, embedding_idx, _ in glove_match)  

Y = [[word2idx[token] for token in headline.split()] for headline in heads]  
len(Y)  

X = [[word2idx[token] for token in d.split()] for d in desc]  
len(X)  


import _pickle as pickle  
with open('_dataset/%s.pkl'%FN,'wb') as fp:  
&emsp;pickle.dump((embedding, idx2word, word2idx, glove_idx2idx),fp,-1)  

import _pickle as pickle  
with open('_dataset/%s.data.pkl'%FN,'wb') as fp:  
&emsp;pickle.dump((X,Y),fp,-1)  


---

Remember 

These commands are for reference copy the commands from original jp-notebook file for execution.
