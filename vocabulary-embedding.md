## Vocabulary Embedding Commands

This file contains commands written in proper sequential manner equivalent to corresponding jupyter notebook file with ipython3 i.e. these are ported commands to python3.

### Command Section  

#### Initialization

`FN = 'vocabulary-embedding'`  

`seed=16`  

`vocab_size = 30000`  

`embedding_dim = 50`  

`lower = False # dont lower case the text`  

`pwd`  

#### Reading tokenized title and content

```
import json  
fndata = '_dataset/sample-1M.jsonl'  
heads = []  
desc = []  
counter = 0  
with open(fndata) as f:  
        for line in f:  
            if counter < 50000 :  
                jdata = json.loads(line)    # for json lines file, loading line by line  
                heads.append(jdata["title"].lower())  
                desc.append(jdata["content"].lower())  
                counter +=1  
```

```
if lower:  
    heads = [h.lower() for h in heads]  
```

```
if lower:
    desc = [h.lower() for h in desc]
```

`len(desc),len(set(desc))`  

`len(heads),len(set(heads))`  

#### Build Vocabulary

```
from collections import Counter  
from itertools import chain  
def get_vocab(lst):  
    vocabcount = Counter(w for txt in lst for w in txt.split())  
    vocab = map(lambda x: x[0], sorted(vocabcount.items(), key=lambda x: -x[1]))  
    return vocab, vocabcount  
```

`vocab, vocabcount = get_vocab(heads+desc)`  

```
import matplotlib.pyplot as plt  
plt.plot([vocabcount[w] for w in vocab]);  
plt.gca().set_xscale("log", nonposx='clip')  
plt.gca().set_yscale("log", nonposy='clip')  
plt.title('word distribution in headlines and discription')  
plt.xlabel('rank')  
plt.ylabel('total appearances');  
```

#### Indexing words

```
empty = 0 # RNN mask of no data  
eos = 1  # end of sentence  
start_idx = eos+1 # first real word  
```

```
def get_idx(vocab, vocabcount):  
    word2idx = dict((word, idx+start_idx) for idx,word in enumerate(vocab))  
    word2idx['<empty>'] = empty  
    word2idx['<eos>'] = eos  
    idx2word = dict((idx,word) for word,idx in word2idx.items())  
    # Important changes in python 3  
    # Removed dict.iteritems(), dict.iterkeys(), and dict.itervalues().  
    # Instead: use dict.items(), dict.keys(), and dict.values() respectively.  
    return word2idx, idx2word  
```

`word2idx, idx2word = get_idx(vocab, vocabcount)`  

#### Word Embedding : Read Glove

`from keras.utils.data_utils import get_file`  

```
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
```

```
glove_name = '/home/user_name   /headlines/glove.6B.50d.txt'  
glove_n_symbols = !wc -l {glove_name}  
glove_n_symbols = int(glove_n_symbols[0].split()[0])  
glove_n_symbols  
```

```
import numpy as np

glove_index_dict = {}  
glove_embedding_weights = np.empty((glove_n_symbols, embedding_dim))  
globale_scale=.1  
with open(glove_name, 'r') as fp:  
    i = 0  
    for l in fp:  
        l = l.strip().split()  
        w = l[0]  
        glove_index_dict[w] = i  
        glove_embedding_weights[i,:] = list(map(float,l[1:]))  
        i += 1  
glove_embedding_weights *= globale_scale  
```

`glove_embedding_weights.std()`

```
for w,i in glove_index_dict.items():  
    w = w.lower()  
    if w not in glove_index_dict:  
        glove_index_dict[w] = i  
```

#### Embedding Matrix

```
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
    w = idx2word[i]  
    g = glove_index_dict.get(w, glove_index_dict.get(w.lower()))  
    if g is None and w.startswith('#'): # glove has no hashtags (I think...)  
        w = w[1:]  
        g = glove_index_dict.get(w, glove_index_dict.get(w.lower()))  
    if g is not None:  
        embedding[i,:] = glove_embedding_weights[g,:]  
        c+=1  
print ('number of tokens, in small vocab, found in glove and copied to embedding', c,c/float(vocab_size))  
```

```
glove_thr = 0.5  
```

```
word2glove = {}  
for w in word2idx:  
    if w in glove_index_dict:  
        g = w  
    elif w.lower() in glove_index_dict:  
        g = w.lower()  
    elif w.startswith('#') and w[1:] in glove_index_dict:  
        g = w[1:]  
    elif w.startswith('#') and w[1:].lower() in glove_index_dict:  
        g = w[1:].lower()  
    else:  
        continue  
    word2glove[w] = g  
```

```
normed_embedding = embedding/np.array([np.sqrt(np.dot(gweight,gweight)) for gweight in embedding])[:,None]
nb_unknown_words = 100
glove_match = []  
for w,idx in word2idx.items():  
    if idx >= vocab_size-nb_unknown_words and w.isalpha() and w in word2glove:  
        gidx = glove_index_dict[word2glove[w]]  
        gweight = glove_embedding_weights[gidx,:].copy()  
        # find row in embedding that has the highest cos score with gweight  
        gweight /= np.sqrt(np.dot(gweight,gweight))  
        score = np.dot(normed_embedding[:vocab_size-nb_unknown_words], gweight)  
        while True:  
            embedding_idx = score.argmax()  
            s = score[embedding_idx]  
            if s < glove_thr:  
                break  
            if idx2word[embedding_idx] in word2glove :  
                glove_match.append((w, embedding_idx, s))  
                break  
            score[embedding_idx] = -1  
glove_match.sort(key = lambda x: -x[2])  
print ('# of glove substitutes found', len(glove_match))  
```

```
for orig, sub, score in glove_match[-10:]:  
    print (score, orig,'=>', idx2word[sub])  
```

`glove_idx2idx = dict((word2idx[w],embedding_idx) for  w, embedding_idx, _ in glove_match)`  

#### Data  

```
Y = [[word2idx[token] for token in headline.split()] for headline in heads]  
len(Y)  
```

```
X = [[word2idx[token] for token in d.split()] for d in desc]  
len(X)  
```

```
import _pickle as pickle  
with open('_dataset/%s.pkl'%FN,'wb') as fp:  
    pickle.dump((embedding, idx2word, word2idx, glove_idx2idx),fp,-1)  
```

```
import _pickle as pickle  
with open('_dataset/%s.data.pkl'%FN,'wb') as fp:  
    pickle.dump((X,Y),fp,-1)  
```

---

#### Note 

* These commands are for reference purpose only, copy the commands from original jp-notebook file for execution with python 3 as kernel.  
* Also, the last two graphs are omitted from this project.  
* The corresponding jupyter notebook contains all the commands above but are executed in jupyter notebook server.
