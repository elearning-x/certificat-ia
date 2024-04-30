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
  title: Text pre-processing and features
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Matthieu LABEAU<br />Chloé CLAVEL</span>
<span>Licence CC BY-NC-ND</span>
</div>

+++

# Lab : Text pre-processing and features

## Objectives

1. Implement a simple way to represent text data - Bag of words / TF-IDF
2. Use the features for classification and sentence similarity tasks with simple models
3. Try to improve results with pre-processing tools from Natural Language Processing

## Necessary dependancies

We will need the following packages:
- The Machine Learning API Scikit-learn : http://scikit-learn.org/stable/install.html
- The Natural Language Toolkit : http://www.nltk.org/install.html

Both are available with Anaconda: https://anaconda.org/anaconda/nltk and https://anaconda.org/anaconda/scikit-learn


```{code-cell} python
import os.path as op
import re 
import numpy as np
import matplotlib.pyplot as plt
```

## I Classification of IMDB Data in sentiment

### I.1 Loading data

We use data from https://ai.stanford.edu/~amaas/data/sentiment/. The archive file has been unzipped on the server and the data are available in **/data/aclImdb**.

We retrieve the textual data in the variable *texts*.

The labels are retrieved in the variable $y$ - it contains *len(texts)* of them: $0$ indicates that the corresponding review is negative while $1$ indicates that it is positive.


```{code-cell} python
from glob import glob
# We get the files from the path: data/aclImdb/train/neg for negative reviews, and data/aclImdb/train/pos for positive reviews
train_filenames_neg = sorted(glob(op.join('data', 'aclImdb', 'train', 'neg', '*.txt')))
train_filenames_pos = sorted(glob(op.join('data', 'aclImdb', 'train', 'pos', '*.txt')))

"""
test_filenames_neg = sorted(glob(op.join('data', 'aclImdb', 'test', 'neg', '*.txt')))
test_filenames_pos = sorted(glob(op.join('data', 'aclImdb', 'test', 'pos', '*.txt')))
"""

# Each files contains a review that consists in one line of text: we put this string in two lists, that we concatenate
train_texts_neg = [open(f, encoding="utf8").read() for f in train_filenames_neg]
train_texts_pos = [open(f, encoding="utf8").read() for f in train_filenames_pos]
train_texts = train_texts_neg + train_texts_pos

"""
test_texts_neg = [open(f, encoding="utf8").read() for f in test_filenames_neg]
test_texts_pos = [open(f, encoding="utf8").read() for f in test_filenames_pos]
test_texts = test_texts_neg + test_texts_pos
"""

# The first half of the elements of the list are string of negative reviews, and the second half positive ones
# We create the labels, as an array of [1,len(texts)], filled with 1, and change the first half to 0
train_labels = np.ones(len(train_texts), dtype=int)
train_labels[:len(train_texts_neg)] = 0.

"""
test_labels = np.ones(len(test_texts), dtype=np.int)
test_labels[:len(test_texts_neg)] = 0.
"""
```


```{code-cell} python
open("data/aclImdb/train/neg/0_3.txt", encoding="utf8").read()
```

**In this lab, the impact of our choice of representations upon our results will also depend on the quantity of data we use:** try to see how changing the parameter ```k``` affects our results !


```{code-cell} python
# This number of documents may be high for most computers: we can select a fraction of them (here, one in k)
# Use an even number to keep the same number of positive and negative reviews
k = 10
train_texts_reduced = train_texts[0::k]
train_labels_reduced = train_labels[0::k]

print('Number of documents:', len(train_texts_reduced))
```

We can use a function from sklearn, ```train_test_split```, to separate data into training and validation sets:


```{code-cell} python
from sklearn.model_selection import train_test_split
```


```{code-cell} python
train_texts_splt, val_texts, train_labels_splt, val_labels = train_test_split(train_texts_reduced, train_labels_reduced, test_size=.2)
```

### I.2 Adapted representation of documents

Our statistical model, like most models applied to textual data, uses counts of word occurrences in a document. Thus, a very convenient way to represent a document is to use a Bag-of-Words (BoW) vector, containing the counts of each word (regardless of their order of occurrence) in the document. 

If we consider the set of all the words appearing in our $T$ training documents, which we note $V$ (Vocabulary), we can create **an index**, which is a bijection associating to each $w$ word an integer, which will be its position in $V$. 

Thus, for a document extracted from a set of documents containing $|V|$ different words, a BoW representation will be a vector of size $|V|$, whose value at the index of a word $w$ will be its number of occurrences in the document. 

We can use the **CountVectorizer** class from scikit-learn to obtain these representations:


```{code-cell} python
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, ClassifierMixin
```


```{code-cell} python
corpus = ['I walked down down the boulevard',
          'I walked down the avenue',
          'I ran down the boulevard',
          'I walk down the city',
          'I walk down the the avenue']
vectorizer = CountVectorizer()

Bow = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names_out())
Bow.toarray()
```

We display the list containing the words ordered according to their index (Note that words of 2 characters or less are not counted).

#### Getting BoW representations

The first thing to do is to turn the review from a string into a list of words. The simplest method is to divide the string according to spaces with the command:
``text.split()``

But we must also be careful to remove special characters that may not have been cleaned up (such as HTML tags if the data was obtained from web pages). Since we're going to count words, we'll have to build a list of tokens appearing in our data. In our case, we'd like to reduce this list and make it uniform (ignore capitalization, punctuation, and the shortest words). 
We will therefore use a function adapted to our needs - but this is a job that we generally don't need to do ourselves, since there are many tools already adapted to most situations. 
For text cleansing, there are many scripts, based on different tools (regular expressions, for example) that allow you to prepare data. The division of the text into words and the management of punctuation is handled in a step called *tokenization*; if needed, a python package like NLTK contains many different *tokenizers*.


```{code-cell} python
# We might want to clean the file with various strategies:
def clean_and_tokenize(text):
    """
    Cleaning a document with:
        - Lowercase        
        - Removing numbers with regular expressions
        - Removing punctuation with regular expressions
        - Removing other artifacts
    And separate the document into words by simply splitting at spaces
    Params:
        text (string): a sentence or a document
    Returns:
        tokens (list of strings): the list of tokens (word units) forming the document
    """        
    # Lowercase
    text = text.lower()
    # Remove numbers
    text = re.sub(r"[0-9]+", "", text)
    # Remove punctuation
    REMOVE_PUNCT = re.compile("[.;:!\'?,\"()\[\]]")
    text = REMOVE_PUNCT.sub("", text)
    # Remove small words (1 and 2 characters)
    text = re.sub(r"\b\w{1,2}\b", "", text)
    # Remove HTML artifacts specific to the corpus we're going to work with
    REPLACE_HTML = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
    text = REPLACE_HTML.sub(" ", text)
    
    tokens = text.split()        
    return tokens

corpus_raw = "I walked down down the boulevard. I walked down the avenue. I ran down the boulevard. I walk down the city. I walk down the the avenue."
print(clean_and_tokenize(corpus_raw))
```

The next function takes as input a list of documents (each in the form of a string) and returns, as in the example using ``CountVectorizer``:
- A vocabulary that associates, to each word encountered, an index
- A matrix, with rows representing documents and columns representing words indexed by the vocabulary. In position $(i,j)$, one should have the number of occurrences of the word $j$ in the document $i$.

The vocabulary, which was in the form of a *list* in the previous example, can be returned in the form of a *dictionary* whose keys are the words and values are the indices. Since the vocabulary lists the words in the corpus without worrying about their number of occurrences, it can be built up using a set (in python).
<div class='alert alert-block alert-info'>
            Code:</div>


```{code-cell} python
def count_words(texts):
    #
    # To complete
    #
    return vocabulary, counts
```


```{code-cell} python
Voc, X = count_words(corpus)
print(Voc)
print(X)
```

Now, if we want to represent text that was not available when building the vocabulary, we will not be able to represent **new words** ! Let's take a look at how CountVectorizer does it:


```{code-cell} python
val_corpus = ['I walked up the street']
Bow = vectorizer.transform(val_corpus)
Bow.toarray()
```

Modify the ```count_words``` function to be able to deal with new documents when given a previously obtained vocabulary ! 
<div class='alert alert-block alert-info'>
            Code:</div>


```{code-cell} python
def count_words(texts, voc = None):
    
    if voc == None:
    #
    # To complete
    #
    else:
    #
    # To complete
    #
    return vocabulary, counts
```

<div class='alert alert-block alert-warning'>
            Question:</div>

Careful: check the memory that the representations are going to use (given the way they are build). What ```CountVectorizer``` argument allows to avoid the issue ? 


```{code-cell} python
voc, train_bow = count_words(train_texts_splt)
print(train_bow.shape)
```


```{code-cell} python
_, val_bow = count_words(val_texts, voc)
print(val_bow.shape)
```


```{code-cell} python
# Create and fit the vectorizer to the training data

```


```{code-cell} python
# Transform the validation data

```

### I.3 Experimentation with classification

We are going to use the scikit-learn ```MultinomialNB```, an implementation of the Naive Bayesian model. Experiment on this model with your own representations. Visualize the results with the following tools, and compare with the representations of ```CountVectorizer```:
<div class='alert alert-block alert-info'>
            Code:</div>


```{code-cell} python
from sklearn.naive_bayes import MultinomialNB
# Fit the model on the training data

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
# Test it on the validation data 

```


```{code-cell} python
# Same with the other representations

```

<div class='alert alert-block alert-warning'>
            Questions:</div>

- Here, what is the naïve hypothesis ? 
- Let us look at the *features* built by the ```vectorizer```. How can we improve them ? 


```{code-cell} python
print(vectorizer.get_feature_names_out()[:100])
```

### I.4 Improving representations: by reweighting and filtering

Mainly, the arguments of the class ```vectorizer``` will allow us to easily change the way our textual data is represented. Let us try to work on our *Bag-of-words* representations:
   
#### Do not take into account words that are too frequent:

You can use the argument ```max_df=1.0``` to change the amount of words taken into account. 

#### Try different granularities:

Rather than just counting words, we can count sequences of words - limited in size, of course. 
We call a sequence of $n$ words a $n$-gram: let's try using 2 and 3-grams (bi- and trigrams).
We can also try to use character sequences instead of word sequences.

We will be interested in the options ```analyze='word'``` and ```ngram_range=(1, 2)``` which we'll change to alter the granularity. 

**Again: using these ways of getting more features from our text will probably have more impact if we do not have much training data to begin with !**

To accelerate experiments, use the ```Pipeline``` tool from scikit-learn. 
<div class='alert alert-block alert-info'>
            Code:</div>


```{code-cell} python
from sklearn.pipeline import Pipeline
```


```{code-cell} python
pipeline_base = Pipeline([
    ('vect', CountVectorizer(max_features=30000, analyzer='word', stop_words=None)),
    ('clf', MultinomialNB()),
])
pipeline_base.fit(train_texts_splt, train_labels_splt)
val_pred = pipeline_base.predict(val_texts)
print(classification_report(val_labels, val_pred))
```


```{code-cell} python

```


```{code-cell} python

```


```{code-cell} python

```

#### Tf-idf:

This is the product of the frequency of the term (TF) and its inverse frequency in documents (IDF).
This method is usually used to measure the importance of a term $i$ in a document $j$ relative to the rest of the corpus, from a matrix of occurrences $ words \times documents$. Thus, for a matrix $\mathbf{T}$ of $|V|$ terms and $D$ documents:
$$\text{TF}(T, w, d) = \frac{T_{w,d}}{\sum_{w'=1}^{|V|} T_{w',d}} $$

$$\text{IDF}(T, w) = \log\left(\frac{D}{|\{d : T_{w,d} > 0\}|}\right)$$

$$\text{TF-IDF}(T, w, d) = \text{TF}(X, w, d) \cdot \text{IDF}(T, w)$$

It can be adapted to our case by considering that the context of the second word is the document. However, TF-IDF is generally better suited to low-density matrices, since it will penalize terms that appear in a large part of the documents. 
<div class='alert alert-block alert-info'>
            Code:</div>


```{code-cell} python
from sklearn.preprocessing import normalize

def tfidf(bow):
    """
    Inverse document frequencies applied to our bag-of-words representations
    """
    # IDF
    
    #
    # To complete ! 
    #
    
    # TF
    
    #
    # To complete ! 
    #
    return tf_idf
```

Experiment with this new representations and compare with the ```TfidfTransformer``` applied on top of ```CountVectorizer```.
<div class='alert alert-block alert-info'>
            Code:</div>


```{code-cell} python

```


```{code-cell} python

```


```{code-cell} python
from sklearn.feature_extraction.text import TfidfTransformer
```


```{code-cell} python

```

## II More pre-processing : custom vocabularies

### II.1 Getting vocabularies

For more flexibility, we will implement separately a function returning the vocabulary. Here we will have to be able to control its size, either by indicating a **maximum number of words**, or a **minimum number of occurrences** to take the words into account. **We add, at the end, an "unknown" word that will replace all the words that do not appear in our "limited" vocabulary**.

<div class='alert alert-block alert-info'>
            Code:</div>


```{code-cell} python
def vocabulary(corpus, count_threshold=5, voc_threshold=10000):
    """    
    Function using word counts to build a vocabulary - can be improved with a second parameter for 
    setting a frequency threshold
    Params:
        corpus (list of strings): corpus of sentences
        count_threshold (int): number of occurences necessary for a word to be included in the vocabulary
        voc_threshold (int): maximum size of the vocabulary. Use "0" to indicate no limit 
    Returns:
        vocabulary (dictionary): keys: list of distinct words across the corpus
                                 values: indexes corresponding to each word sorted by frequency   
        vocabulary_word_counts (dictionary): keys: list of distinct words across the corpus
                                             values: corresponding counts of words in the corpus
    """
    word_counts = {}
    for sent in corpus:
    #
    # To complete: count word frequencies
    #
    filtered_word_counts = ... # Filter according to count_threhshold        
    sorted_words = ... # Extract the words according to frequency
    filtered_words = ... # Remove the words above voc-threshold
    words = ... # Add UNK
    vocabulary = ... # Create vocabulary from "words"
    return vocabulary, {word: filtered_word_counts.get(word, 0) for word in vocabulary}
```


```{code-cell} python
# Example for testing:

corpus = ['I walked down down the boulevard',
          'I walked down the avenue',
          'I ran down the boulevard',
          'I walk down the city',
          'I walk down the the avenue']

voc, counts = vocabulary(corpus, count_threshold = 3)
print(voc)
print(counts)

# We expect something like this:
# (In this example, we don't count 'UNK' unknown words, but you can if you want to. 
# How useful it may be depends on the data -> we will use the counts later with word2vec, keep that in mind) 
#  {'down': 0, 'the': 1, 'i': 2, 'UNK': 3}
#  {'down': 6, 'the': 6, 'i': 5, 'UNK': 0}

voc, counts = vocabulary(corpus)
print(voc)
print(counts)

# We expect something like this:
#  {'down': 0, 'the': 1, 'i': 2, 'walked': 3, 'boulevard': 4, 'avenue': 5, 'walk': 6, 'ran': 7, 'city': 8, 'UNK': 9}
#  {'down': 6, 'the': 6, 'i': 5, 'walked': 2, 'boulevard': 2, 'avenue': 2, 'walk': 2, 'ran': 1, 'city': 1, 'UNK': 0}
```

#### Quick study of the data

We would like to get an idea of what's in these film reviews. So we'll get the vocabulary (in full) and represent the frequencies of the words, in order (be careful, you'll have to use a logarithmic scale): we should find back Zipf's law. This will give us an idea of the size of the vocabulary we will be able to choose: it's a matter of making a compromise between the necessary resources (size of the objects in memory) and the amount of information we can get from them (rare words can bring a lot of information, but it's difficult to learn good representations of them, because they are rare!).  

<div class='alert alert-block alert-info'>
            Code:</div>


```{code-cell} python
# We would like to display the curve of word frequencies given their rank (index) in the vocabulary
vocab, word_counts = vocabulary(train_texts)
#
#  To fill in !
#

# We can for example use the function plt.scatter()
plt.figure(figsize=(20,5))
plt.title('Word counts versus rank')
#
#  To fill in !
#
plt.yscale('log')
plt.show()

# We would like to know how much of the data is represented by the 'k' most frequent words
print('Vocabulary size: %i' % len(vocab))
print('Part of the corpus by taking the "x" most frequent words ?')
#
#  To fill in !-
#
```

<div class='alert alert-block alert-warning'>
            Questions:</div>
            
Word2vec's implementation cuts the vocabulary size by using **only words with at least 5 occurences**, by default. What vocabulary size would it give here ? Does it seem like a good compromise, looking at the graph ? 


```{code-cell} python

```

### II.2 With pre-processing tools from NLTK

We are now going to pre-process our textual data. **Note that this still will only be useful if we do not have a lot of training data to begin with !**

#### Stemming 

Allows to go back to the root of a word: you can group different words around the same root, which facilitates generalization. Use:
```from nltk import SnowballStemmer```


```{code-cell} python
from nltk import SnowballStemmer
stemmer = SnowballStemmer("english")
```

**Example:**


```{code-cell} python
words = ['singers', 'cat', 'generalization', 'philosophy', 'psychology', 'philosopher']
for word in words:
    print('word : %s ; stemmed : %s' %(word, stemmer.stem(word)))#.decode('utf-8'))))
```

**Data transformation:**
<div class='alert alert-block alert-info'>
            Code:</div>


```{code-cell} python
def stem(X): 
    X_stem = []
    #
    # To complete ! 
    #
    return X_stem
```

#### Part of speech tags

To generalize, we can also use the Part of Speech (POS) of the words, which will allow us to filter out information that is potentially not useful to the model. We will retrieve the POS of the words using the functions:
```pos_tag```


```{code-cell} python
import nltk
from nltk import pos_tag
```

**Example:**


```{code-cell} python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

pos_tag(clean_and_tokenize(('I am Sam')))
```

**Data transformation:** only keep nouns, verbs, adverbs, and adjectives (```['NN', 'VB', 'ADJ', 'RB']```) for our model.
<div class='alert alert-block alert-info'>
            Code:</div>


```{code-cell} python
def pos_tag_filter(X, good_tags=['NN', 'VB', 'ADJ', 'RB']):
    X_pos = []
    #
    # To complete ! 
    #
    return X_pos
```

### II.3 Application

<div class='alert alert-block alert-warning'>
            Questions:</div>

Re-draw the Zipf distribution of our data **after reducing their vocabulary with these functions**. How is it affected ? How do you think it could affect results here ?         
        
<div class='alert alert-block alert-info'>
            Code:</div>        


```{code-cell} python

```

## III Semantic Textual Similarity

Semantic Textual Similarity (STS) measures the degree of semantic equivalence between two texts. As an NLP task, it typically consists in determining, for two sentences $s_1$ and $s_2$, how similar they are in meaning. Systems must output a continous score $p$ between, for example, $0$ (completely unrelated) and $1$ (meaning-equivalent). For example, for these two unrelated sentences, $p$ should be close to $0$:

$s_1$: The black dog is running through the snow.

$s_2$: A race car driver is driving his car through the mud.

This task poses a regression problem, so instead of a Naïve Bayes classifier we will use sklearn's ```LinearRegression``` model. The feature we will feed as input to the regressor - and which will represent a pair of sentences - is the cosine distance between their vector representations.


```{code-cell} python
import os
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
```

### III.1 Loading the data


We will use data from the STSBenchmark. The dataset comes with a pre-defined train/dev/test split. For simplicity we will use the train and dev portions together as our training set.

Download the data from here http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz and decompress it:


```{code-cell} python
def load_data(path):
    data = {'train':dict(), 'test':dict()}
    for fn in os.listdir(path):
        if fn.endswith(".csv"):
            with open(op.join(path, fn)) as f:
                subset = fn[:-4].split("-")[1]        
                if subset == "dev":
                    subset = "train"        
                data[subset]['data'] = []
                data[subset]['scores'] = []
                for l in f:          
                    l = l.strip().split("\t")          
                    data[subset]['data'].append((l[5],l[6]))
                    data[subset]['scores'].append(float(l[4]) / 5) # mapping the score to the 0-1 range 
    return data

sts_dataset = load_data(path="stsbenchmark")
```

Using the code of the ```load_data``` function, get an insight into the structure of the dataset. Print the first few examples ($s_1$, $s_2$ and the score) and the number of examples in the dataset.
<div class='alert alert-block alert-info'>
            Code:</div>


```{code-cell} python
### Having a look at the data...
print("Some examples from the dataset:")
for i in range(5):
    ...

print("\nNumber of sentence pairs by subset:")

```

### III.2 Experimentation

The goal is now to:
- Obtain vector representations of the sentences
- Compute their cosine distance
- Fit a regressor to predict the similarity score
- Predict the scores for the test data and compare it to the gold scores using Spearman correlation (```spearmanr```)

To obtain vector representations, use the functions we previously created and used (```count_words``` and ```CountVectorizer```), and experiment with:
- Classic bow representations
- Tf-idf vectors

<div class='alert alert-block alert-info'>
            Code:</div>


```{code-cell} python
# Obtain all the sentences in only one list (no pairs) to be able to get the vocabulary

```


```{code-cell} python
# Obtain the voc/Fit the vectorizer on the training data

```


```{code-cell} python
# Transform train and test data. Calculate the cosine between the representations of each sentence pair

```


```{code-cell} python
from sklearn.linear_model import LinearRegression
```


```{code-cell} python
# Train a linear regression model, make predictions on test set, evaluate it using spearman's r

```


```{code-cell} python
# Repeat the last two steps with tf-idf instead

```
