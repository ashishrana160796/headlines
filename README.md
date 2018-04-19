# Automatically generate headlines to short articles

### Aim : This summarizer will be acting as a submodule in NLPChatbot to summarize a conversation with user and extract out context for meaningful replies.

__Procedure :__

* To conduct training only on partial dataset( 100k _approx._).
* Port this project to Python 3.5 instead of python 2.7. 
* Making a suitable plugin from it for above NLPChatbot.
* _glove.6B.50d.txt_ is used for training for reduced the training time.
 
__Current System State :__

* tensorflow 1.7 gpu
* keras 2.1.5
* python 3.5
 
---
## How to run
### Software
* The code is running with [jupyter notebook](http://jupyter.org/)
* Install [Tensorflow](https://www.tensorflow.org/)
* `pip install python-Levenshtein`

### Data
It is assumed that you already have training and test data.
The data is made from many examples (I'm using 684K examples),
each example is made from the text
from the start of the article, which I call description (or `desc`),
and the text of the original headline (or `head`).
The texts should be already tokenized and the tokens separated by spaces.

Once you have the data ready save it in a python pickle file as a tuple:
`(heads, descs, keywords)` were `heads` is a list of all the head strings,
`descs` is a list of all the article strings in the same order and length as `heads`.
I ignore the `keywords` information so you can place `None`.

### Build a vocabulary of words
The [vocabulary-embedding](./vocabulary-embedding.ipynb)
notebook describes how a dictionary is built for the tokens and how
an initial embedding matrix is built from [GloVe](http://nlp.stanford.edu/projects/glove/)

### Train a model
[train](./train.ipynb) notebook describes how a model is trained on the data using [Tensorflow](https://www.tensorflow.org/)

### Use model to generate new headlines
[predict](./predict.ipynb) generate headlines by the trained model and
showes the attention weights used to pick words from the description.
The text generation includes a feature which was
not described in the original paper, it allows for words that are outside
the training vocabulary to be copied from the description to the generated headline.

## Examples of headlines generated
Good (cherry picking) examples of headlines generated
![cherry picking of generated headlines](./cherry_picking.png)
![cherry picking of generated headlines](./cherry_picking1.png)

## Examples of attention weights
![attention weights](./attention_weights.png)

---

The MIT License (MIT)

Copyright (c) 2016 Ehud Ben-Reuven

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
