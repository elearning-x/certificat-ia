---
jupytext:
  cell_metadata_filter: all, -hidden, -heading_collapsed, -run_control, -trusted
  notebook_metadata_filter: all, -jupytext.text_representation.jupytext_version, -jupytext.text_representation.format_version,
    -language_info.version, -language_info.codemirror_mode.version, -language_info.codemirror_mode,
    -language_info.file_extension, -language_info.mimetype, -toc
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
language_info:
  name: python
  nbconvert_exporter: python
  pygments_lexer: ipython3
nbhosting:
  title: Classification correction
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Matthieu LABEAU<br/ >Chloé CLAVEL</span>
<span>Licence CC BY-NC-ND</span>
</div>

+++

# Text classification with Pytorch

The goal of this lab are to:
- Implement an 'handmade' model of text classification with word embeddings,
- Learn how to use Pytorch for treating textual data, 
- Implementing neural classification models with Pytorch.

Besides ```torch```, we will use ```gensim``` to obain word embeddings, and ```scikit-learn``` for simple classification models.  


```{code-cell} python
import os.path as op
import re 
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
```

## I - Simple classifier on top of dense representations 

### I.1 Dataset

We're going to work with the **20NewsGroup** data. This dataset is available in ```scikit-learn```, you can find all relevant information in the [documentation](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html).


```{code-cell} python
from sklearn.datasets import fetch_20newsgroups
```


```{code-cell} python
# Import training data
ng_train = fetch_20newsgroups(subset='train',
                              remove=('headers', 'footers', 'quotes')
                              )
```


```{code-cell} python
# Let's look at what is in this object
pprint(dir(ng_train))
```

    ['DESCR', 'data', 'filenames', 'target', 'target_names']


```{code-cell} python
# Let's look at the categories
pprint(ng_train.target_names)
```

    ['alt.atheism',
     'comp.graphics',
     'comp.os.ms-windows.misc',
     'comp.sys.ibm.pc.hardware',
     'comp.sys.mac.hardware',
     'comp.windows.x',
     'misc.forsale',
     'rec.autos',
     'rec.motorcycles',
     'rec.sport.baseball',
     'rec.sport.hockey',
     'sci.crypt',
     'sci.electronics',
     'sci.med',
     'sci.space',
     'soc.religion.christian',
     'talk.politics.guns',
     'talk.politics.mideast',
     'talk.politics.misc',
     'talk.religion.misc']



```{code-cell} python
# .. and the data itself
pprint(ng_train.data[0])
print("Target: ", ng_train.target_names[ng_train.target[0]])
```

    ('I was wondering if anyone out there could enlighten me on this car I saw\n'
     'the other day. It was a 2-door sports car, looked to be from the late 60s/\n'
     'early 70s. It was called a Bricklin. The doors were really small. In '
     'addition,\n'
     'the front bumper was separate from the rest of the body. This is \n'
     'all I know. If anyone can tellme a model name, engine specs, years\n'
     'of production, where this car is made, history, or whatever info you\n'
     'have on this funky looking car, please e-mail.')
    Target:  rec.autos


The dataset can be rather difficult as it is; especially, some categories are very close to each other. We can simplify the task by using the higher-level categorisation of the newsgroups, thanks to the following function:


```{code-cell} python
def aggregate_labels(label):
    # comp
    if label in [1,2,3,4,5]:
        new_label = 0
    # rec
    if label in [7,8,9,10]:
        new_label = 1
    # sci
    if label in [11,12,13,14]:
        new_label = 2
    # misc 
    if label in [6]:
        new_label = 3
    # pol
    if label in [16,17,18]:
        new_label = 4
    # rel
    if label in [0,15,19]:
        new_label = 5
    return new_label
```

We will first need to apply some pre-processing. Choose your tokenizer and the processing you estimate appropriate. Careful, the data is not always clean and the messages are sometimes short: hence, applying pre-processing and tokenization can easily return an empty list of words. **Be careful to remove documents that are empty !**
<div class='alert alert-block alert-info'>
            Code:</div>


```{code-cell} python
from nltk import word_tokenize
import nltk
nltk.download('punkt')
```


```{code-cell} python
# Pre-processing 
ng_train_text = [ng_train.data[i] for i in range(len(ng_train.data)) if len(word_tokenize(ng_train.data[i])) > 0]
ng_train_labels = [aggregate_labels(ng_train.target[i]) for i in range(len(ng_train.data)) if len(word_tokenize(ng_train.data[i])) > 0]
```


```{code-cell} python
ng_test = fetch_20newsgroups(subset='test',
                             remove=('headers', 'footers', 'quotes')
                            )

ng_test_text = [ng_test.data[i] for i in range(len(ng_test.data)) if len(word_tokenize(ng_test.data[i])) > 0]
ng_test_labels = [aggregate_labels(ng_test.target[i]) for i in range(len(ng_test.data)) if len(word_tokenize(ng_test.data[i])) > 0]
```

### I.2 Get a vocabulary.

Now that the data is cleaned, the first step we will follow is to pick a common vocabulary that we will use for every model we create in this lab. **Use the code of the previous lab to create a vocabulary.** As in the previous lab, we will have to be able to control its size, either by indicating a maximum number of words, or a minimum number of occurrences to take the words into account. Again, we add, at the end, an "unknown" word that will replace all the words that do not appear in our "limited" vocabulary. 
<div class='alert alert-block alert-info'>
            Code:</div>


```{code-cell} python
def vocabulary(corpus, voc_threshold = None):
    """    
    Function using word counts to build a vocabulary - can be improved with a second parameter for 
    setting a frequency threshold
    Params:
        corpus (list of list of strings): corpus of sentences
        voc_threshold (int): maximum size of the vocabulary 
    Returns:
        vocabulary (dictionary): keys: list of distinct words across the corpus
                                 values: indexes corresponding to each word sorted by frequency        
    """
    word_counts = {}
    for sent in corpus:
        for word in word_tokenize(sent):
            if word not in word_counts:
                word_counts[word] = 0
            word_counts[word] += 1           
    words = sorted(word_counts.keys(), key=word_counts.get, reverse=True)
    if voc_threshold is not None:
        words = words[:voc_threshold] 
    words = words + ['UNK'] 
    vocabulary = {words[i] : i for i in range(len(words))}
    return vocabulary, {word: word_counts.get(word, 0) for word in vocabulary}
```

<div class='alert alert-block alert-warning'>
            Question:</div>
            
What do you think is the **appropriate vocabulary size here** ? Would any further pre-processing make sense ? Motivate your answer. 


```{code-cell} python
vocab, word_counts = vocabulary(ng_train_text)
rank_counts = {w:[vocab[w], word_counts[w]] for w in vocab}
rank_counts_array = np.array(list(rank_counts.values()))

plt.figure(figsize=(20,5))
plt.title('Word counts versus rank')
plt.yscale('log')
plt.ylim(0, 1e6)
plt.scatter(rank_counts_array[:,0], rank_counts_array[:,1])
plt.show()

print('Vocabulary size: %i' % len(vocab))
print('Part of the corpus by taking the "x" most frequent words:')
for i in range(5000, len(vocab), 5000):
    print('%i : %.2f' % (i, np.sum(rank_counts_array[:i, 1]) / np.sum(rank_counts_array[:,1]) ))
```

    /home/matthieu/anaconda3/envs/TPs_NLP/lib/python3.7/site-packages/ipykernel_launcher.py:8: UserWarning: Attempted to set non-positive bottom ylim on a log-scaled axis.
    Invalid limit will be ignored.
      



    
![png](media/Part_2/2.12/Lab_Part_2_Classification_Correction_18_1.png)
    


    Vocabulary size: 161713
    Part of the corpus by taking the "x" most frequent words:
    5000 : 0.85
    10000 : 0.89
    15000 : 0.91
    20000 : 0.93
    25000 : 0.94
    30000 : 0.94
    35000 : 0.95
    40000 : 0.95
    45000 : 0.96
    50000 : 0.96
    55000 : 0.96
    60000 : 0.97
    65000 : 0.97
    70000 : 0.97
    75000 : 0.97
    80000 : 0.97
    85000 : 0.97
    90000 : 0.98
    95000 : 0.98
    100000 : 0.98
    105000 : 0.98
    110000 : 0.98
    115000 : 0.98
    120000 : 0.99
    125000 : 0.99
    130000 : 0.99
    135000 : 0.99
    140000 : 0.99
    145000 : 0.99
    150000 : 1.00
    155000 : 1.00
    160000 : 1.00


Before creating the vocabulary, put aside some training data for a **validation set** ! 


```{code-cell} python
from sklearn.model_selection import train_test_split
train_texts_splt, val_texts, train_labels_splt, val_labels = train_test_split(ng_train_text, ng_train_labels, test_size=.2)
```


```{code-cell} python
vocab_cut, word_counts_cut = vocabulary(train_texts_splt, 10000)
```

### I.3 Getting a representation: commonly used algorithms with ```gensim```

The idea here is to define a set of representations $({w_{i}})_{i=1}^{V}$, of predefined dimension $d$ (here, we will work with $d = 300$), for all the words $i$ of the vocabulary $V$ - then **train** these representations to match what we want. 

#### Glove

The objective defined by Glove ([Pennington et al. (2014)](http://www.aclweb.org/anthology/D/D14/D14-1162.pdf)) is to learn from the vectors $w_{i}$ and $w_{k}$ so that their scalar product corresponds to the logarithm of their **Pointwise Mutual Information**: 


$$ w_{i}^\top w_{k} = (PMI(w_{i}, w_{k}))$$


In the article, this objective is carefully justified by a reasoning about the operations one wants to perform with these vectors and the properties they should have - in particular, symmetry between rows and columns (see the article for more details).  
The final goal obtained is the following, where $M$ is the co-occurrence matrix:


$$\sum_{i, j=1}^{|V|} f\left(M_{ij}\right)
  \left(w_i^\top w_j + b_i + b_j - \log M_{ij}\right)^2$$
  
 
Here, $f$ is a *scaling* function that reduces the importance of the most frequent co-occurrence counts: 


$$f(x) 
\begin{cases}
(x/x_{\max})^{\alpha} & \textrm{if } x < x_{\max} \\
1 & \textrm{otherwise}
\end{cases}$$


Usually, we choose $\alpha=0.75$ and $x_{\max} = 100$, although these parameters may need to be changed depending on the data. The following code uses the ```gensim``` API to retrieve pre-trained representations (word embeddings take space - a long loading time is expected).


```{code-cell} python
import gensim.downloader as api
loaded_glove_model = api.load("glove-wiki-gigaword-300")
```

We can extract the embedding matrix this way, and check its size:


```{code-cell} python
loaded_glove_embeddings = loaded_glove_model.vectors
print(loaded_glove_embeddings.shape)
```

    (400000, 300)


We can see that there are $400,000$ words represented, and that the embeddings are of size $300$. We define a function that returns, from the loaded model, the vocabulary and the embedding matrix according to the structures we used before. We add, here again, an unknown word "UNK" in case there are words in our data that are not part of the $400,000$ words represented here. 


```{code-cell} python
def get_glove_voc_and_embeddings(glove_model):
    voc = {word : index for word, index in enumerate(glove_model.index_to_key)}
    voc['UNK'] = len(voc)
    embeddings = glove_model.vectors
    return voc, embeddings
```


```{code-cell} python
loaded_glove_voc, loaded_glove_embeddings = get_glove_voc_and_embeddings(loaded_glove_model)
```

To be able to merge these $400.000$ words with those that are in our vocabulary, we can create a specific function that will extract the representations of the words that are in our vocabulary and return a matrix of the appropriate size:


```{code-cell} python
def get_glove_adapted_embeddings(glove_model, input_voc):
    keys = {i: glove_model.key_to_index.get(w, None) for w, i in input_voc.items()}
    index_dict = {i: key for i, key in keys.items() if key is not None}
    embeddings = np.zeros((len(input_voc),glove_model.vectors.shape[1]))
    for i, ind in index_dict.items():
        embeddings[i] = glove_model.vectors[ind]
    return embeddings
```

This function takes as input the model loaded using the Gensim API, as well as a vocabulary we created ourselves, and returns the embedding matrix from the loaded model, for the words in our vocabulary and in the right order.



```{code-cell} python
GloveEmbeddings = get_glove_adapted_embeddings(loaded_glove_model, vocab_cut)
```


```{code-cell} python
print(GloveEmbeddings.shape)
```

    (10001, 300)


#### Word2Vec


The basic skip-gram model estimates the probabilities of a pair of words $(i, j)$ to appear together in data:

$$P(j \mid i) = \frac{\exp(w_{i} c_{j})}{\sum_{j'\in V}\exp(w_{i} c_{j'})}$$


where $w_{i}$ is the lign vector (of the word) $i$ and $c_{j}$ is the column vector (of a context word) $j$. The objective is to minimize the following quantity:


$$ -\sum_{i=1}^{m} \sum_{k=1}^{|V|} \textbf{1}\{o_{i}=k\} \log \frac{\exp(w_{i} c_{k})}{\sum_{j=1}^{|V|} \exp(w_{i} c_{j})}$$


where $V$ is the vocabulary.
The inputs $w_{i}$ are the representations of the words, which are updated during training, and the output is an *one-hot* $o$ vector, which contains only one $1$ and $0$. For example, if `good` is the 47th word in the vocabulary, the output $o$ for an example or `good` is the word to predict will consist of $0$s everywhere except $1$ in the 47th position of the vector. `good` will be the word to predict when the input $w$ is a word in its context.
We therefore obtain this output with standard softmax - we add a bias term $b$ .


$$ o = \textbf{softmax}(w_{a}C + b)$$


If we use the set of representations for the whole vocabulary (the matrix $W$) as input, we get:


$$ O = \textbf{softmax}(WC + b)$$


and so we come back to the central idea of all our methods: we seek to obtain word representations from co-occurrence counts. Here, we train the parameters contained in $W$ and $C$, two matrices representing the words in reduced dimension (300) so that their scalar product is as close as possible to the co-occurrences observed in the data, using a maximum likelihood objective.

#### Skip-gram with negative sampling

The training of the skip-gram model implies to calculate a sum on the whole vocabulary, because of the **softmax**. As soon as the size of the vocabulary increases, it becomes impossible to compute. In order to make the calculations faster, we change the objective and use the method of *negative sampling* (or, very close to it, the *noise contrastive estimation*).


If we note $\mathcal{D}$ the data set and we note $\mathcal{D}'$ a set of pairs of words that are **not** in the data (and that in practice, we draw randomly), the objective is:


$$\sum_{i, j \in \mathcal{D}}-\log\sigma(w_{i}c_{j}) + \sum_{i, j \in \mathcal{D}'}\log\sigma(w_{i}c_{j})$$


where $\sigma$ is the sigmoid activation function $\frac{1}{1 + \exp(-x)}$.
A common practice is to generate pairs from $\mathcal{D}'$ in proportion to the frequencies of the words in the training data (the so-called unigram distribution):


$$P(w) = \frac{\textbf{T}(w)^{0.75}}{\sum_{w'\in V} \textbf{T}(w')}$$


Although different, this new objective function is a sufficient approximation of the previous one, and is based on the same principle. Much research has been done on this objective: for example, [Levy and Golberg 2014](http://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization) shows that the objective calculates the PMI matrix shifted by a constant value. One can also see [Cotterell et al. 2017](https://aclanthology.coli.uni-saarland.de/papers/E17-2028/e17-2028) for an interpretation of the algorithm as a variant of PCA.

We will use the ```gensim``` library for its implementation of word2vec in python. We'll have to make a specific use of it, since we want to keep the same vocabulary as before: we'll first create the class, then get the vocabulary we generated above. 


```{code-cell} python
from gensim.models import Word2Vec

model = Word2Vec(vector_size=300,
                 window=5,
                 null_word=len(word_counts_cut),
                 epochs=30)
model.build_vocab_from_freq(word_counts_cut)
```

The model takes as input a **list of list of words**: you need to tokenize the data beforehand. 
In this case, you also need to indicate to the model the number of examples it should train with.
<div class='alert alert-block alert-info'>
            Code:</div>


```{code-cell} python
ng_train_text_tokenized = []
ex = 0
for sent in train_texts_splt:
    tok_sent = word_tokenize(sent)
    ng_train_text_tokenized.append(tok_sent)
    ex += len(tok_sent)
```


```{code-cell} python
model.train(ng_train_text_tokenized, total_examples=ex, epochs=30, report_delay=1)
```




    (44154398, 74092890)




```{code-cell} python
W2VEmbeddings = model.wv.vectors
print(W2VEmbeddings.shape)
```

    (10001, 300)


### I.4 Application to classification:

We will now use these representations for classification.
The basic model will be constructed in two steps:
- A function to obtain vector representations of criticism, from text, vocabulary, and vector representations of words. Such a function (to be completed below) will associate to each word of a review its embeddings, and create the representation for the whole sentence by summing these embeddings.
- A classifier will take these representations as input and make a prediction. To achieve this, we can first use logistic regression ```LogisticRegression``` from ```scikit-learn``` 

<div class='alert alert-block alert-info'>
            Code:</div>


```{code-cell} python
def sentence_representations(texts, vocabulary, embeddings, np_func=np.mean):
    """
    Represent the sentences as a combination of the vector of its words.
    Parameters
    ----------
    texts : a list of sentences   
    vocabulary : dict
        From words to indexes of vector.
    embeddings : Matrix containing word representations
    np_func : function (default: np.sum)
        A numpy matrix operation that can be applied columnwise, 
        like `np.mean`, `np.sum`, or `np.prod`. 
    Returns
    -------
    np.array, dimension `(len(texts), embeddings.shape[1])`            
    """
    representations = []
    for text in texts:
        indexes = np.array([vocabulary.get(w,len(vocabulary)-1) for w in word_tokenize(text)])
        sentrep = np_func(embeddings[indexes], axis=0)
        representations.append(sentrep)
    representations = np.array(representations)    
    return representations
```


```{code-cell} python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Example with Glove Embeddings
train_rep = sentence_representations(train_texts_splt, vocab_cut, GloveEmbeddings)
val_rep = sentence_representations(val_texts, vocab_cut, GloveEmbeddings)
clf = LogisticRegression().fit(train_rep, train_labels_splt)
print(clf.score(val_rep, val_labels))
```

    0.7271901951883795


    /home/matthieu/anaconda3/envs/TPs_NLP/lib/python3.7/site-packages/sklearn/linear_model/_logistic.py:818: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG,



```{code-cell} python
train_rep_w2v = sentence_representations(train_texts_splt, vocab_cut, W2VEmbeddings)
val_rep_w2v = sentence_representations(val_texts, vocab_cut, W2VEmbeddings)
clf_w2v = LogisticRegression().fit(train_rep_w2v, train_labels_splt)
print(clf_w2v.score(val_rep_w2v, val_labels))
```

    0.6177939173853836


    /home/matthieu/anaconda3/envs/TPs_NLP/lib/python3.7/site-packages/sklearn/linear_model/_logistic.py:818: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG,


<div class='alert alert-block alert-warning'>
            Question:</div>

- Why can we expect that the results obtained with embeddings extracted from representations pre-trained with Gl0ve are much better than word2vec ? What would be a 'fair' way to compare Gl0ve with word2vec ?


```{code-cell} python

```

# II - Text classification with Pytorch

The goal of this second part of the lab is double: an introduction to using Pytorch for treating textual data, and implementing neural classification models that we can apply to IMDB data - and then compare it to the models implemented previously. 


```{code-cell} python
import torch
import torch.nn as nn
```

### II.1 A (very small) introduction to pytorch

Pytorch Tensors are very similar to Numpy arrays, with the added benefit of being usable on GPU. For a short tutorial on various methods to create tensors of particular types, see [this link](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py).
The important things to note are that Tensors can be created empty, from lists, and it is very easy to convert a numpy array into a pytorch tensor, and inversely.


```{code-cell} python
a = torch.LongTensor(5)
b = torch.LongTensor([5])

print(a)
print(b)
```

    tensor([139667514253904, 139667514253968, 139667548501968, 139667514278944,
            139667514379360])
    tensor([5])



```{code-cell} python
a = torch.FloatTensor([2])
b = torch.FloatTensor([3])

print(a + b)
```

    tensor([5.])


The main interest in us using Pytorch is the ```autograd``` package. ```torch.Tensor```objects have an attribute ```.requires_grad```; if set as True, it starts to track all operations on it. When you finish your computation, can call ```.backward()``` and all the gradients are computed automatically (and stored in the ```.grad``` attribute).

One way to easily cut a tensor from the computational once it is not needed anymore is to use ```.detach()```.
More info on automatic differentiation in pytorch on [this link](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py).


```{code-cell} python
x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

# Build a computational graph.
y = w * x + b    # y = 2 * x + 3

# Compute gradients.
y.backward()

# Print out the gradients.
print(x.grad)    # x.grad = 2 
print(w.grad)    # w.grad = 1 
print(b.grad)    # b.grad = 1 
```

    tensor(2.)
    tensor(1.)
    tensor(1.)



```{code-cell} python
x = torch.randn(10, 3)
y = torch.randn(10, 2)

# Build a fully connected layer.
linear = nn.Linear(3, 2)
for name, p in linear.named_parameters():
    print(name)
    print(p)

# Build loss function - Mean Square Error
criterion = nn.MSELoss()

# Forward pass.
pred = linear(x)

# Compute loss.
loss = criterion(pred, y)
print('Initial loss: ', loss.item())

# Backward pass.
loss.backward()

# Print out the gradients.
print ('dL/dw: ', linear.weight.grad) 
print ('dL/db: ', linear.bias.grad)
```

    weight
    Parameter containing:
    tensor([[ 0.4410, -0.1242,  0.1920],
            [ 0.3673, -0.2458, -0.0321]], requires_grad=True)
    bias
    Parameter containing:
    tensor([-0.5655,  0.0166], requires_grad=True)
    Initial loss:  1.3672114610671997
    dL/dw:  tensor([[ 0.2794,  0.2498,  0.5792],
            [-0.0240, -0.2971, -0.9171]])
    dL/db:  tensor([-0.6966, -0.0036])



```{code-cell} python
# You can perform gradient descent manually, with an in-place update ...
linear.weight.data.sub_(0.01 * linear.weight.grad.data)
linear.bias.data.sub_(0.01 * linear.bias.grad.data)

# Print out the loss after 1-step gradient descent.
pred = linear(x)
loss = criterion(pred, y)
print('Loss after one update: ', loss.item())
```

    Loss after one update:  1.3484481573104858



```{code-cell} python
# Use the optim package to define an Optimizer that will update the weights of the model.
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

# By default, gradients are accumulated in buffers( i.e, not overwritten) whenever .backward()
# is called. Before the backward pass, we need to use the optimizer object to zero all of the
# gradients.
optimizer.zero_grad()
loss.backward()

# Calling the step function on an Optimizer makes an update to its parameters
optimizer.step()

# Print out the loss after the second step of gradient descent.
pred = linear(x)
loss = criterion(pred, y)
print('Loss after two updates: ', loss.item())
```

    Loss after two updates:  1.3302618265151978


### II.2 Tools for data processing 

```torch.utils.data.Dataset``` is an abstract class representing a dataset. Your custom dataset should inherit ```Dataset``` and override the following methods:
- ```__len__``` so that ```len(dataset)``` returns the size of the dataset.
- ```__getitem__``` to support the indexing such that ```dataset[i]``` can be used to get the i-th sample

Here is a toy example: 


```{code-cell} python
toy_corpus = ['I walked down down the boulevard',
              'I walked down the avenue',
              'I ran down the boulevard',
              'I walk down the city',
              'I walk down the the avenue']

toy_categories = [0, 0, 1, 0, 0]
```


```{code-cell} python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    # A pytorch dataset class for holding data for a text classification task.
    def __init__(self, data, categories):
        # Upon creating the Dataset object, store the data in an attribute
        # Split the text data and labels from each other
        self.X, self.Y = [], []
        for x, y in zip(data, categories):
            # We will propably need to preprocess the data - have it done in a separate method
            # We do it here because we might need corpus-wide info to do the preprocessing 
            # For example, cutting all examples to the same length
            self.X.append(self.preprocess(x))
            self.Y.append(y)
                
    # Method allowing you to preprocess data                      
    def preprocess(self, text):
        text_pp = text.lower().strip()
        return text_pp
    
    # Overriding the method __len__ so that len(CustomDatasetName) returns the number of data samples                     
    def __len__(self):
        return len(self.Y)
   
    # Overriding the method __getitem__ so that CustomDatasetName[i] returns the i-th sample of the dataset                      
    def __getitem__(self, idx):
           return self.X[idx], self.Y[idx]
```


```{code-cell} python
toy_dataset = CustomDataset(toy_corpus, toy_categories)
```


```{code-cell} python
print(len(toy_dataset))
for i in range(len(toy_dataset)):
    print(toy_dataset[i])
```

    5
    ('i walked down down the boulevard', 0)
    ('i walked down the avenue', 0)
    ('i ran down the boulevard', 1)
    ('i walk down the city', 0)
    ('i walk down the the avenue', 0)


```torch.utils.data.DataLoader``` is what we call an iterator, which provides very useful features:
- Batching the data
- Shuffling the data
- Load the data in parallel using multiprocessing workers.
and can be created very simply from a ```Dataset```. Continuing on our simple example: 


```{code-cell} python
toy_dataloader = DataLoader(toy_dataset, batch_size = 2, shuffle = True)
```


```{code-cell} python
for e in range(3):
    print("Epoch:" + str(e))
    for x, y in toy_dataloader:
        print("Batch: " + str(x) + "; labels: " + str(y))  
```

    Epoch:0
    Batch: ('i walked down the avenue', 'i walk down the the avenue'); labels: tensor([0, 0])
    Batch: ('i walk down the city', 'i ran down the boulevard'); labels: tensor([0, 1])
    Batch: ('i walked down down the boulevard',); labels: tensor([0])
    Epoch:1
    Batch: ('i ran down the boulevard', 'i walk down the the avenue'); labels: tensor([1, 0])
    Batch: ('i walked down the avenue', 'i walked down down the boulevard'); labels: tensor([0, 0])
    Batch: ('i walk down the city',); labels: tensor([0])
    Epoch:2
    Batch: ('i walked down the avenue', 'i walk down the the avenue'); labels: tensor([0, 0])
    Batch: ('i ran down the boulevard', 'i walked down down the boulevard'); labels: tensor([1, 0])
    Batch: ('i walk down the city',); labels: tensor([0])


#### Data processing of a text dataset

Now, we would like to apply what we saw to our case, and **create a specific class** ```TextClassificationDataset``` **inheriting** ```Dataset``` that will:
- Create a vocabulary from the data (same as above)
- Preprocess the data using this vocabulary, adding whatever we need for our pytorch model
- Have a ```__getitem__``` method that allows us to use the class with a ```Dataloader``` to easily build batches.


```{code-cell} python
from torch.nn import functional as F
import random

from torch.nn.utils.rnn import pad_sequence
```

We will now to create a ```TextClassificationDataset``` and a ```Dataloader``` for the training data, the validation data, and the testing data.

**OLD - NOT NECESSARY ANYMORE**

We will implement ourselves a function that will help us split the data in three, according to proportions we give in input.


```{code-cell} python
# Create a function allowing you to simply shuffle then split the filenames and categories into the desired
# proportions for a training, validation and testing set. 
def get_splits(x, y, splits):
    """
    The idea is to use an index list as reference:
    Indexes = [0 1 2 3 4 5 6 7 8 9]
    To shuffle it randomly:
    Indexes = [7 1 5 0 2 9 8 6 4 3]
    We need 'splits' to contain 2 values. Assuming those are = (0.8, 0.1), we'll have:
    Train_indexes = [7 1 5 0 2 9 8 6]
    Valid_indexes = [4]
    Test_indexes = [3]
    """
    # Create an index list and shuffle it
    l = len(x)
    indexes = list(range(l))
    random.shuffle(indexes)
    
    # Find the two indexes we'll use to cut the lists
    sep_0 = int(splits[0]*l)
    sep_1 = int((splits[1]+splits[0])*l)
    
    # Do the cutting (careful: you can't use a list as index for a list - this only works with tensors)
    # (you need to use list comprehensions - or go through numpy)
    train_x, train_y = [x[i] for i in indexes[:sep_0]], [y[i] for i in indexes[:sep_0]]
    valid_x, valid_y = [x[i] for i in indexes[sep_0:sep_1]], [y[i] for i in indexes[sep_0:sep_1]]
    test_x, test_y = [x[i] for i in indexes[sep_1:]], [y[i] for i in indexes[sep_1:]]
    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)
```


```{code-cell} python
# Choose the training, validation, testing splits
# splits = (0.8, 0.1)
# (train_f, train_c), (valid_f, valid_c), (test_f, test_c) = get_splits(filenames, categories, splits)
```

We can now implement our ```TextClassificationDataset``` class, that we will build from:
- A list of documents: ```data```
- A list of the corresponding categories: ```categories```
We will add three optional arguments:
- First, a way to input a vocabulary (so that we can re-use the training vocabulary on the validation and training ```TextClassificationDataset```). By default, the value of the argument is ```None```.
- In order to work with batches, we will need to have sequences of the same size. That can be done via **padding** but we will still need to limit the size of documents (to avoid having batches of huge sequences that are mostly empty because of one very long documents) to a ```max_length```. Let's put it to 100 by default.
- Lastly, a ```min_freq``` that indicates how many times a word must appear to be taken in the vocabulary. 

The idea behind **padding** is to transform a list of pytorch tensors (of maybe different length) into a two dimensional tensor - which we can see as a batch. The size of the first dimension is the one of the longest tensor - and other are **padded** with a chosen symbol: here, we choose 0. 

**Careful: the symbol 0 is then reserved for padding. That means the vocabulary must begin at 1 !** 


```{code-cell} python
tensor_1 = torch.LongTensor([1, 4, 5])
tensor_2 = torch.LongTensor([2])
tensor_3 = torch.LongTensor([6, 7])
```


```{code-cell} python
tensor_padded = pad_sequence([tensor_1, tensor_2, tensor_3], batch_first=True, padding_value = 0)
print(tensor_padded)
```

    tensor([[1, 4, 5],
            [2, 0, 0],
            [6, 7, 0]])


<div class='alert alert-block alert-info'>
            Code:</div>


```{code-cell} python
class TextClassificationDataset(Dataset):
    def __init__(self, data, categories, vocab = None, max_length = 200, voc_threshold = 10000):
        # Get all the data in a list
        self.data = data
        # Set the maximum length we will keep for the sequences
        self.max_length = max_length
        # Allow to import a vocabulary (for valid/test datasets, that will use the training vocabulary)
        if vocab is not None:
            self.word2idx, self.idx2word = vocab
        else:
            # If no vocabulary imported, build it (and reverse)
            self.word2idx, self.idx2word = self.build_vocab(self.data, voc_threshold)
        
        # Part to extend - we tokenize each document into a list of words, which we transform in indexes (without forgetting UNK)
        # Each document is now a list of indexes of words, that we cut at the max_length and embed into a pytorch tensor
        # We then apply padding to the list of tensors, choosing the value that represents the pad, and that the first dimension if the batch
        self.tensor_data = pad_sequence(
            [torch.LongTensor([self.word2idx.get(w,self.word2idx['UNK']) for w in word_tokenize(self.data[i])][:self.max_length]) for i in range(len(self.data))],
            batch_first=True,
            padding_value = 0
        )
        self.tensor_y = torch.LongTensor(categories)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # The iterator just gets one particular example with its category
        # The dataloader will take care of the shuffling and batching
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.tensor_data[idx], self.tensor_y[idx] 
    
    def build_vocab(self, corpus, voc_threshold):
        # Careful, here we need to shift the indexes by 1 to put the padding symbol to 0
        word_counts = {}
        for sent in corpus:
            for word in word_tokenize(sent):
                if word not in word_counts:
                    word_counts[word] = 0
                word_counts[word] += 1           
        words = sorted(word_counts.keys(), key=word_counts.get, reverse=True)
        if voc_threshold is not None:
            words = words[:voc_threshold] 
        words = words + ['UNK'] 
        vocabulary = {words[i] : i+1 for i in range(len(words))}
        return vocabulary, {word: word_counts.get(word, 0) for word in vocabulary}
    
    def get_vocab(self):
        # A simple way to get the training vocab when building the valid/test 
        return self.word2idx, self.idx2word
```


```{code-cell} python
training_dataset = TextClassificationDataset(train_texts_splt, train_labels_splt)
training_word2idx, training_idx2word = training_dataset.get_vocab()
```


```{code-cell} python
valid_dataset = TextClassificationDataset(val_texts, val_labels, (training_word2idx, training_idx2word))
test_dataset = TextClassificationDataset(ng_test_text, ng_test_labels, (training_word2idx, training_idx2word))
```


```{code-cell} python
training_dataloader = DataLoader(training_dataset, batch_size = 200, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size = 25)
test_dataloader = DataLoader(test_dataset, batch_size = 25)
```


```{code-cell} python
print(valid_dataset[1])
```

    (tensor([   94,   846,     8,   973,    67,   102,  5384,    40, 10001, 10001,
                1,    47,    10,   770,     8, 10001,  9145,    16,    10,   338,
             2944,   359,   196,   448,     2,    45, 10001,    14,   234,   430,
               54,     3, 10001,   262,     8,     3,  8582,    14,    39,     2,
            10001,     9, 10001, 10001,    11,  1262,   506,   179,    24,   353,
              196,    54,    14,    49,  5251,   138,   173,     2,   283,   846,
               14,   173,   894, 10001,     9,   253, 10001,    11,    54,    42,
               14,    49,  5251,   430,     3,  1154,    56,    65,     2,   126,
                3,   750,   141,   730,    68,   846,    67,   118,   555,   894,
            10001,   816,    40,     3, 10001, 10001,     1,    54,     3,  1646,
                8,     3,  4717,  1467,    35,    49,  1141,     7,    30,  3488,
               24,   359,   730,     2,   235,    10,   770,     8,     3,  1154,
                1,    68,   846,    14,   373,   332, 10001,     2, 10001,    16,
               66,   659,   182,   267,    14,    58, 10001,     2,  1294,   689,
            10001,     7,  1154,    14,    18, 10001,    67,   173,  1375,     9,
              403,  6581,    12,   539,    40, 10001,    56,     3,  1242,     8,
                3, 10001,    28,  1986,     8,  1778,    28, 10001,    12,  5315,
            10001,    11,     2,    45, 10001,    14,   377,  8583,   427,    68,
              846,    44,  1587,    67,    39,  2982,    83,   701,     8,  1487,
                7, 10001,    23,     2,    45,  1314,  9448,    14,    18,     3]), tensor(2))



```{code-cell} python
example_batch = next(iter(training_dataloader))
print(example_batch[0].size())
print(example_batch[1].size())
```

    torch.Size([200, 200])
    torch.Size([200])


### II.3 A simple averaging model

Now, we will implement in Pytorch what we did in the previous TP: a simple averaging model. For each model we will implement, we need to create a class which inherits from ```nn.Module``` and redifine the ```__init__``` method as well as the ```forward``` method.


```{code-cell} python
import torch.optim as optim
```

<div class='alert alert-block alert-info'>
            Code:</div>


```{code-cell} python
class AveragingModel(nn.Module):
    
    def __init__(self, embedding_dim, vocabulary_size, categories_num):
        super().__init__()
        # Create an embedding object. Be careful to padding - you need to increase the vocabulary size by one !
        # Look into the arguments of the nn.Embedding class
        self.embeddings = nn.Embedding(vocabulary_size + 1, embedding_dim, padding_idx = 0)
        # Create a linear layer that will transform the mean of the embeddings into classification scores
        self.linear = nn.Linear(embedding_dim, categories_num)
        
    def forward(self, inputs):
        # Remember: the inpts are written as Batch_size * seq_length * embedding_dim
        # First, take the mean of the embeddings of the document
        x = self.embeddings(inputs).mean(dim = 1)
        o = self.linear(x).squeeze()
        return o
```


```{code-cell} python
model = AveragingModel(300, len(training_word2idx), max(ng_train_labels)+1)
# Create an optimizer
opt = optim.Adam(model.parameters(), lr=0.0025, betas=(0.9, 0.999))
# The criterion is a cross entropy loss based on logits, 
# meaning that the softmax is integrated into the criterion
criterion = nn.CrossEntropyLoss()
```

<div class='alert alert-block alert-info'>
            Code:</div>


```{code-cell} python
# Implement a training function, which will train the model with the corresponding optimizer and criterion,
# with the appropriate dataloader, for one epoch.

def train_epoch(model, opt, criterion, dataloader):
    model.train()
    losses = []
    for i, (x, y) in enumerate(dataloader):
        opt.zero_grad()
        # (1) Forward
        pred = model(x)
        # (2) Compute diff
        loss = criterion(pred, y)
        # (3) Compute gradients
        loss.backward()
        # (4) update weights
        opt.step()        
        losses.append(loss.item())
        # Count the number of correct predictions in the batch - here, you'll need to use the softmax
        num_corrects = (torch.argmax(torch.softmax(pred, dim=1), dim=1, keepdim=False).data == (y.data)).float().sum().data
        acc = 100.0 * num_corrects/len(y)
        
        if (i%20 == 0):
            print("Batch " + str(i) + " : training loss = " + str(loss.item()) + "; training acc = " + str(acc.item()))
    return losses
```

<div class='alert alert-block alert-info'>
            Code:</div>


```{code-cell} python
# Same for the evaluation ! We don't need the optimizer here. 

def eval_model(model, criterion, evalloader):
    model.eval()
    total_epoch_loss = 0
    total_epoch_acc = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(evalloader):
            pred = model(x)
            loss = criterion(pred, y)
            num_corrects = (torch.argmax(torch.softmax(pred, dim=1), dim=1, keepdim=False).data == (y.data)).float().sum().data
            acc = 100.0 * num_corrects/len(y)
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss/(i+1), total_epoch_acc/(i+1)
```


```{code-cell} python
# A function which will help you execute experiments rapidly - with a early_stopping option when necessary.

def experiment(model, opt, criterion, num_epochs = 10, early_stopping = True):
    train_losses = []
    if early_stopping: 
        best_valid_loss = 10. 
    print("Beginning training...")
    for e in range(num_epochs):
        print("Epoch " + str(e+1) + ":")
        train_losses += train_epoch(model, opt, criterion, training_dataloader)
        valid_loss, valid_acc = eval_model(model, criterion, valid_dataloader)
        print("Epoch " + str(e+1) + " : Validation loss = " + str(valid_loss) + "; Validation acc = " + str(valid_acc))
        if early_stopping:
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
            else:
                print("Early stopping.")
                break  
    test_loss, test_acc = eval_model(model, criterion, test_dataloader)
    print("Epoch " + str(e+1) + " : Test loss = " + str(test_loss) + "; Test acc = " + str(test_acc))
    return train_losses
```


```{code-cell} python
train_losses = experiment(model, opt, criterion)
```

    Beginning training...
    Epoch 1:
    Batch 0 : training loss = 1.8056769371032715; training acc = 11.0
    Batch 20 : training loss = 1.654463529586792; training acc = 38.5
    Batch 40 : training loss = 1.6475051641464233; training acc = 38.0
    Epoch 1 : Validation loss = 1.600007311681683; Validation acc = 42.741573033707866
    Epoch 2:
    Batch 0 : training loss = 1.5613807439804077; training acc = 41.0
    Batch 20 : training loss = 1.490339756011963; training acc = 52.0
    Batch 40 : training loss = 1.3795934915542603; training acc = 60.5
    Epoch 2 : Validation loss = 1.4110725033149292; Validation acc = 53.03370786516854
    Epoch 3:
    Batch 0 : training loss = 1.4064757823944092; training acc = 53.5
    Batch 20 : training loss = 1.2307296991348267; training acc = 63.0
    Batch 40 : training loss = 1.1756876707077026; training acc = 64.0
    Epoch 3 : Validation loss = 1.2158572955077953; Validation acc = 61.79775280898876
    Epoch 4:
    Batch 0 : training loss = 1.1780277490615845; training acc = 62.5
    Batch 20 : training loss = 1.1238799095153809; training acc = 63.0
    Batch 40 : training loss = 0.9984109401702881; training acc = 70.0
    Epoch 4 : Validation loss = 1.074562084139063; Validation acc = 65.61797752808988
    Epoch 5:
    Batch 0 : training loss = 1.0082972049713135; training acc = 71.5
    Batch 20 : training loss = 0.9112349152565002; training acc = 73.5
    Batch 40 : training loss = 0.9321964383125305; training acc = 71.5
    Epoch 5 : Validation loss = 0.9602339267730713; Validation acc = 69.66292134831461
    Epoch 6:
    Batch 0 : training loss = 0.8480947613716125; training acc = 75.5
    Batch 20 : training loss = 0.705613911151886; training acc = 81.0
    Batch 40 : training loss = 0.8641168475151062; training acc = 74.5
    Epoch 6 : Validation loss = 0.8769700031601981; Validation acc = 72.76404494382022
    Epoch 7:
    Batch 0 : training loss = 0.7420291304588318; training acc = 78.5
    Batch 20 : training loss = 0.7013176679611206; training acc = 81.5
    Batch 40 : training loss = 0.7244132161140442; training acc = 77.5
    Epoch 7 : Validation loss = 0.817697957660375; Validation acc = 74.42696629213484
    Epoch 8:
    Batch 0 : training loss = 0.6258180737495422; training acc = 82.0
    Batch 20 : training loss = 0.5705943703651428; training acc = 82.5
    Batch 40 : training loss = 0.6574228405952454; training acc = 77.0
    Epoch 8 : Validation loss = 0.7667487581794181; Validation acc = 76.2247191011236
    Epoch 9:
    Batch 0 : training loss = 0.5832177996635437; training acc = 88.0
    Batch 20 : training loss = 0.5707685351371765; training acc = 85.5
    Batch 40 : training loss = 0.502348005771637; training acc = 86.5
    Epoch 9 : Validation loss = 0.729882724164577; Validation acc = 77.1685393258427
    Epoch 10:
    Batch 0 : training loss = 0.6020035743713379; training acc = 85.0
    Batch 20 : training loss = 0.556253969669342; training acc = 85.0
    Batch 40 : training loss = 0.6165981888771057; training acc = 80.0
    Epoch 10 : Validation loss = 0.7011782161975175; Validation acc = 77.66292134831461
    Epoch 10 : Test loss = 0.7561805836169793; Test acc = 74.28869704419029



```{code-cell} python
import matplotlib.pyplot as plt
plt.plot(train_losses)
```




    [<matplotlib.lines.Line2D at 0x7f06d096a290>]




    
![png](media/Part_2/2.12/Lab_Part_2_Classification_Correction_97_1.png)
    


### II.4 Initializing with pre-trained embeddings: 

Now, we would like to integrate pre-trained word embeddings into our model ! However, we need to not forget to add a vector for the padding symbol.


```{code-cell} python
def get_glove_adapted_embeddings(glove_model, input_voc):
    keys = {i: glove_model.key_to_index.get(w, None) for w, i in input_voc.items()}
    index_dict = {i: key for i, key in keys.items() if key is not None}
    # Important change here: add one supplementary word for padding
    embeddings = np.zeros((len(input_voc)+1,glove_model.vectors.shape[1])) 
    for i, ind in index_dict.items():
        embeddings[i] = glove_model.vectors[ind]
    return embeddings

GloveEmbeddings = get_glove_adapted_embeddings(loaded_glove_model, training_word2idx)
```


```{code-cell} python
print(GloveEmbeddings.shape)
```

    (10002, 300)


Here, implement a ```PretrainedAveragingModel``` very similar to the previous model, using the ```nn.Embedding``` method ```from_pretrained()``` to initialize the embeddings from a numpy array. Use the ```requires_grad_``` method to specify if the model must fine-tune the embeddings or not ! 
<div class='alert alert-block alert-info'>
            Code:</div>


```{code-cell} python
class PretrainedAveragingModel(nn.Module):
    
    def __init__(self, embedding_dim, categories_num, embeddings, fine_tuning=False):
        super().__init__()
        # Careful to padding
        self.embeddings = nn.Embedding.from_pretrained(embeddings, padding_idx = 0)
        self.embeddings.requires_grad_(fine_tuning)
        self.linear = nn.Linear(embedding_dim, categories_num)
        
    def forward(self, inputs):
        # Batch_size * seq_length * embedding_dim
        x = self.embeddings(inputs).mean(dim = 1)
        o = self.linear(x).squeeze()
        return o
```

<div class='alert alert-block alert-warning'>
            Questions:</div>
            
- What are the results **with and without fine-tuning of embeddings imported from GloVe** ? Explain them.
- Use the ```sklearn``` function from the previous lab to analyze these results in more details. 


```{code-cell} python
model_pre_trained = PretrainedAveragingModel(300, max(ng_train_labels)+1, torch.FloatTensor(GloveEmbeddings), True)
opt_pre_trained = optim.Adam(model_pre_trained.parameters(), lr=0.0025, betas=(0.9, 0.999))
```


```{code-cell} python
train_losses = experiment(model_pre_trained, opt_pre_trained, criterion)
```

    Beginning training...
    Epoch 1:
    Batch 0 : training loss = 1.8014189004898071; training acc = 7.0
    Batch 20 : training loss = 1.6760108470916748; training acc = 31.0
    Batch 40 : training loss = 1.641151785850525; training acc = 43.0
    Epoch 1 : Validation loss = 1.5733913571647045; Validation acc = 57.03370786516854
    Epoch 2:
    Batch 0 : training loss = 1.517319917678833; training acc = 64.5
    Batch 20 : training loss = 1.4054211378097534; training acc = 62.0
    Batch 40 : training loss = 1.3016257286071777; training acc = 65.5
    Epoch 2 : Validation loss = 1.3103618889712216; Validation acc = 63.235955056179776
    Epoch 3:
    Batch 0 : training loss = 1.2680141925811768; training acc = 70.5
    Batch 20 : training loss = 1.135811448097229; training acc = 69.0
    Batch 40 : training loss = 0.9759698510169983; training acc = 74.5
    Epoch 3 : Validation loss = 1.0824292442771826; Validation acc = 68.35955056179775
    Epoch 4:
    Batch 0 : training loss = 0.9863284230232239; training acc = 73.5
    Batch 20 : training loss = 0.9778892993927002; training acc = 72.5
    Batch 40 : training loss = 0.9139924645423889; training acc = 72.0
    Epoch 4 : Validation loss = 0.9339342559321543; Validation acc = 71.32584269662921
    Epoch 5:
    Batch 0 : training loss = 0.8493244647979736; training acc = 74.5
    Batch 20 : training loss = 0.7613001465797424; training acc = 81.0
    Batch 40 : training loss = 0.6605966687202454; training acc = 84.0
    Epoch 5 : Validation loss = 0.8336341796296366; Validation acc = 74.38202247191012
    Epoch 6:
    Batch 0 : training loss = 0.6796736717224121; training acc = 83.0
    Batch 20 : training loss = 0.6506392955780029; training acc = 81.5
    Batch 40 : training loss = 0.6760450601577759; training acc = 80.0
    Epoch 6 : Validation loss = 0.7664273552010569; Validation acc = 76.71910112359551
    Epoch 7:
    Batch 0 : training loss = 0.6293626427650452; training acc = 83.5
    Batch 20 : training loss = 0.6368820071220398; training acc = 87.0
    Batch 40 : training loss = 0.5842316150665283; training acc = 84.5
    Epoch 7 : Validation loss = 0.7155481275547756; Validation acc = 77.3932584269663
    Epoch 8:
    Batch 0 : training loss = 0.5178633332252502; training acc = 87.5
    Batch 20 : training loss = 0.5154820084571838; training acc = 87.0
    Batch 40 : training loss = 0.4626648426055908; training acc = 88.5
    Epoch 8 : Validation loss = 0.6771946867530265; Validation acc = 78.65168539325843
    Epoch 9:
    Batch 0 : training loss = 0.39506515860557556; training acc = 92.5
    Batch 20 : training loss = 0.4915657937526703; training acc = 87.5
    Batch 40 : training loss = 0.39308875799179077; training acc = 91.5
    Epoch 9 : Validation loss = 0.6484492076246926; Validation acc = 79.37078651685393
    Epoch 10:
    Batch 0 : training loss = 0.4486679136753082; training acc = 89.5
    Batch 20 : training loss = 0.3816889226436615; training acc = 93.0
    Batch 40 : training loss = 0.4055980443954468; training acc = 87.0
    Epoch 10 : Validation loss = 0.6250987471489424; Validation acc = 80.0
    Epoch 10 : Test loss = 0.6848261360422336; Test acc = 77.56434451679321



```{code-cell} python
model_pre_trained_light = PretrainedAveragingModel(300, max(ng_train_labels)+1, torch.FloatTensor(GloveEmbeddings), False)
opt_pre_trained_light = optim.Adam(model_pre_trained_light.parameters(), lr=0.0025, betas=(0.9, 0.999))
```


```{code-cell} python
train_losses = experiment(model_pre_trained_light, opt_pre_trained_light, criterion)
```

    Beginning training...
    Epoch 1:
    Batch 0 : training loss = 1.7777479887008667; training acc = 27.0
    Batch 20 : training loss = 1.664321780204773; training acc = 24.5
    Batch 40 : training loss = 1.6422474384307861; training acc = 32.0
    Epoch 1 : Validation loss = 1.6427893504667819; Validation acc = 36.52434454071388
    Epoch 2:
    Batch 0 : training loss = 1.5950062274932861; training acc = 41.0
    Batch 20 : training loss = 1.628419041633606; training acc = 44.5
    Batch 40 : training loss = 1.5602498054504395; training acc = 52.0
    Epoch 2 : Validation loss = 1.5743365502089597; Validation acc = 48.5692883609386
    Epoch 3:
    Batch 0 : training loss = 1.5509861707687378; training acc = 52.0
    Batch 20 : training loss = 1.5307269096374512; training acc = 50.0
    Batch 40 : training loss = 1.4656916856765747; training acc = 55.0
    Epoch 3 : Validation loss = 1.5179538070485834; Validation acc = 51.49063667554534
    Epoch 4:
    Batch 0 : training loss = 1.521815538406372; training acc = 49.5
    Batch 20 : training loss = 1.499091386795044; training acc = 49.0
    Batch 40 : training loss = 1.4733179807662964; training acc = 57.0
    Epoch 4 : Validation loss = 1.4713368817661585; Validation acc = 50.90636701262399
    Epoch 5:
    Batch 0 : training loss = 1.4627695083618164; training acc = 51.0
    Batch 20 : training loss = 1.3932199478149414; training acc = 59.5
    Batch 40 : training loss = 1.3735933303833008; training acc = 56.0
    Epoch 5 : Validation loss = 1.4300944992665494; Validation acc = 54.142322068803765
    Epoch 6:
    Batch 0 : training loss = 1.4383292198181152; training acc = 54.0
    Batch 20 : training loss = 1.4053057432174683; training acc = 56.0
    Batch 40 : training loss = 1.3501964807510376; training acc = 57.0
    Epoch 6 : Validation loss = 1.3943496674634097; Validation acc = 55.1760299339723
    Epoch 7:
    Batch 0 : training loss = 1.3248145580291748; training acc = 60.5
    Batch 20 : training loss = 1.373378038406372; training acc = 53.0
    Batch 40 : training loss = 1.298525094985962; training acc = 62.0
    Epoch 7 : Validation loss = 1.3626371595296967; Validation acc = 56.25468161936556
    Epoch 8:
    Batch 0 : training loss = 1.353847622871399; training acc = 56.5
    Batch 20 : training loss = 1.3314327001571655; training acc = 55.5
    Batch 40 : training loss = 1.322548270225525; training acc = 56.0
    Epoch 8 : Validation loss = 1.3367631113931033; Validation acc = 56.20973779914085
    Epoch 9:
    Batch 0 : training loss = 1.3274952173233032; training acc = 60.5
    Batch 20 : training loss = 1.2332799434661865; training acc = 64.0
    Batch 40 : training loss = 1.3575161695480347; training acc = 55.0
    Epoch 9 : Validation loss = 1.3112601966000674; Validation acc = 56.65917600138803
    Epoch 10:
    Batch 0 : training loss = 1.3259413242340088; training acc = 52.0
    Batch 20 : training loss = 1.2546297311782837; training acc = 62.0
    Batch 40 : training loss = 1.2313669919967651; training acc = 62.5
    Epoch 10 : Validation loss = 1.289680217089278; Validation acc = 56.70411982161276
    Epoch 10 : Test loss = 1.2850911639656224; Test acc = 56.077896000989064


### II.5 With a LSTM model


```{code-cell} python
# Create a toy example of LSTM: 
lstm = nn.LSTM(3, 3)  # Input dim is 3, output dim is 3
inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5

# LSTMs expect inputs having 3 dimensions:
# - The first dimension is the temporal dimension, along which we (in our case) have the different words
# - The second dimension is the batch dimension, along which we stack the independant batches
# - The third dimension is the feature dimension, along which are the features of the vector representing the words

# In our toy case, we have inputs and outputs containing 3 features (third dimension !)
# We created a sequence of 5 different inputs (first dimension !)
# We don't use batch (the second dimension will have one lement)

# We need an initial hidden state, of the right sizes for dimension 2/3, but with only one temporal element:
# Here, it is:
hidden = (torch.randn(1, 1, 3),
          torch.randn(1, 1, 3))
# Why do we create a tuple of two tensors ? Because we use LSTMs: remember that they use two sets of weights,
# and two hidden states (Hidden state, and Cell state).
# If you don't remember, read: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
# If we used a classic RNN, we would simply have:
# hidden = torch.randn(1, 1, 3)

# The naive way of applying a lstm to inputs is to apply it one step at a time, and loop through the sequence
for i in inputs:
    # After each step, hidden contains the hidden states (remember, it's a tuple of two states).
    out, hidden = lstm(i.view(1, 1, -1), hidden)
    
# Alternatively, we can do the entire sequence all at once.
# The first value returned by LSTM is all of the Hidden states throughout the sequence.
# The second is just the most recent Hidden state and Cell state (you can compare the values)
# The reason for this is that:
# "out" will give you access to all hidden states in the sequence, for each temporal step
# "hidden" will allow you to continue the sequence and backpropagate later, with another sequence
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # Re-initialize
out, hidden = lstm(inputs, hidden)
print(out)
print(hidden)
```

    tensor([[[-0.0853,  0.1664, -0.1331]],
    
            [[-0.0083, -0.0006,  0.0682]],
    
            [[ 0.0535, -0.0982,  0.0098]],
    
            [[ 0.0320, -0.1369, -0.0140]],
    
            [[ 0.0783, -0.0549, -0.1097]]], grad_fn=<StackBackward0>)
    (tensor([[[ 0.0783, -0.0549, -0.1097]]], grad_fn=<StackBackward0>), tensor([[[ 0.2730, -0.1829, -0.3100]]], grad_fn=<StackBackward0>))


### Creating our own LSTM Model

We'll implement now a LSTM model, taking the same inputs and also outputing a score for the sentence.

<div class='alert alert-block alert-info'>
            Code:</div>


```{code-cell} python
# Models are usually implemented as custom nn.Module subclass
# We need to redefine the __init__ method, which creates the object
# We also need to redefine the forward method, which transform the input into outputs

class LSTMModel(nn.Module):
    def __init__(self, embedding_dim, vocabulary_size, hidden_dim, categories_num, embeddings=None, fine_tuning=False):
        super(LSTMModel, self).__init__()
        if embeddings is None:
            self.embeddings = nn.Embedding(vocabulary_size + 1, embedding_dim, padding_idx = 0)
        else: 
            self.embeddings = nn.Embedding.from_pretrained(embeddings)
            self.embeddings.requires_grad_(fine_tuning)
        # Create the LSTM layers !
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, 1, batch_first=True)
        self.linear = nn.Linear(hidden_dim, categories_num)       

    def forward(self, inputs):
        # Process the input
        x = self.embeddings(inputs)
        # Apply the LSTMs
        h, (h_n, h_c) = self.rnn(x, None)
        o = self.linear(h[:, -1, :]).squeeze()
        return o
```


```{code-cell} python
recurrent_model = LSTMModel(300, len(training_word2idx), 100, max(ng_train_labels)+1)
opt_recurrent = optim.Adam(recurrent_model.parameters(), lr=0.0025, betas=(0.9, 0.999))
```


```{code-cell} python
train_losses = experiment(recurrent_model, opt_recurrent, criterion)
```

    Beginning training...
    Epoch 1:
    Batch 0 : training loss = 1.8124706745147705; training acc = 5.5
    Batch 20 : training loss = 1.7800287008285522; training acc = 20.0
    Batch 40 : training loss = 1.7207763195037842; training acc = 27.5
    Epoch 1 : Validation loss = 1.696341006943349; Validation acc = 26.95131083284871
    Epoch 2:
    Batch 0 : training loss = 1.7246785163879395; training acc = 30.0
    Batch 20 : training loss = 1.625372290611267; training acc = 27.0
    Batch 40 : training loss = 1.6985799074172974; training acc = 24.5
    Epoch 2 : Validation loss = 1.6872910823714866; Validation acc = 27.805243417118373
    Epoch 3:
    Batch 0 : training loss = 1.6409430503845215; training acc = 32.0
    Batch 20 : training loss = 1.6253302097320557; training acc = 34.0
    Batch 40 : training loss = 1.653839111328125; training acc = 31.5
    Epoch 3 : Validation loss = 1.6831378494755607; Validation acc = 27.715355776668936
    Epoch 4:
    Batch 0 : training loss = 1.6017647981643677; training acc = 37.5
    Batch 20 : training loss = 1.6821937561035156; training acc = 31.0
    Batch 40 : training loss = 1.643072247505188; training acc = 32.0
    Epoch 4 : Validation loss = 1.7121983565641252; Validation acc = 27.520599236649073
    Early stopping.
    Epoch 4 : Test loss = 1.705926923214779; Test acc = 27.048383862492166


<div class='alert alert-block alert-warning'>
            Questions:</div>
            
- What do you see with a simple application of LSTMs ? 
- What do you think may be happening in this case ? 
