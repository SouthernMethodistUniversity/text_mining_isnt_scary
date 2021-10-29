---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.10.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Data Isn't Scary: Introduction to Text Mining


## Setup


These are the basic libraries we will use in for data manipulation (pandas) and math functions (numpy). We will add more libraries as we need them.

As a best practice, it is a good idea to load all your libraries in a single cell at the top of a notebook, however for the purposes of this tutorial we will load them as we go.

```python
import pandas as pd
import numpy as np
```

Load a data file into a pandas DataFrame.

This tutorial was designed around using sets of data you have yourselves in a form like a CSV, TSV, or TXT file.

To proceed, you will need to have this file downloaded and in the same folder as this notebook. Alternatively you can put the full path to the file.  Typically, your program will look for the file with the name you specified in the folder that contains your program unless you give the program a path to follow.

For this particular lesson, I have made some choices about how I read in the file so that we get a structure that is interesting to analyze.

```python
with open("monster_mash.txt") as f:
    file = f.read().splitlines()

lyrics=[]
verse=[]
for el in file:
    if el!="":
        verse.append(el)
    else:
        lyrics.append("; ".join(verse))
        verse = []
lyrics.append("; ".join(verse))
```

```python
data = pd.DataFrame(lyrics,columns=["Lyrics"]).reset_index()
```

```python
data.head()
```

## Counting Words and Characters


The first bit of analysis we might want to do is to count the number of words in one piece of data. To do this we will add a column called *wordcount* and write an operation that applies a function to every row of the column.

Unpacking this piece of code, *len(str(x).split(" ")*, tells us what is happening.

For the content of cell *x*, convert it to a string, *str()*, then split that string into pieces at each space, *split()*.

The result of that is a list of all the words in the text and then we can count the length of that list, *len()*.

```python
data['wordcount'] = data['Lyrics'].apply(lambda x: len(str(x).split(" ")))
data[['Lyrics','wordcount']].head()
```

We can do something similar to count the number of characters in the data chunk, including spaces. If you wanted to exclude whitespaces, you could take the list we made above, join it together and count the length of the resulting string.

```python
data = data.fillna("No Information Provided") #If some of our data is missing, this will replace the blank entries. This is only necessary in some cases
```

```python
data['char_count'] = data['Lyrics'].str.len() ## this also includes spaces, to do it without spaces, you could use something like this: "".join()
data[['Lyrics','char_count']].head()
```

Now we want to calculate the average word length in the data.

Let's define a function that will do that for us:

```python
def avg_word(sentence):
    words = sentence.split()
    return (sum(len(word) for word in words)/len(words))
```

We can now apply that function to all the data chunks and save that in a new column.

```python
data['avg_word'] = data['Lyrics'].apply(lambda x: avg_word(x))
data[['Lyrics','avg_word']].head()
```

We can then sort by the average word length.

```python
data[['Lyrics','avg_word']].sort_values(by='avg_word', ascending=True).head()
```

# Processing Text


A major component of doing analysis on text is the cleaning of the text prior to the analysis.

Though this process destroys some elements of the text (sentence structure, for example), it is often necessary in order to describe a text analytically. Depending on your choice of cleaning techniques, some elements might be preserved better than others if that is of importance to your analysis.


## Cleaning Up Words


This series of steps aims to clean up and standardize the text itself. This generally consists of removing common elements such as stopwords and punctuation but can be expanded to more detailed removals.


### Lowercase


Here we enforce that all of the text is lowercase. This makes it easier to match cases and sort words.

Notice we are assigning our modified column back to itself. This will save our modifications to our DataFrame

```python
data['Lyrics'] = data['Lyrics'].apply(lambda x: " ".join(x.lower() for x in x.split()))
data['Lyrics'].head()
```

### Remove Punctuation


Here we remove all punctuation from the data. This allows us to focus on the words only as well as assist in matching.

```python
data['Lyrics'] = data['Lyrics'].str.replace('[^\w\s]','')
data['Lyrics'].head()
```

### Remove Stopwords


Stopwords are words that are commonly used and do little to aid in the understanding of the content of a text. There is no universal list of stopwords and they vary on the style, time period and media from which your text came from.  Typically, people choose to remove stopwords from their data, as it adds extra clutter while the words themselves provide little to no insight as to the nature of the data.  For now, we are simply going to count them to get an idea of how many there are.

For this tutorial, we will use the standard list of stopwords provided by the Natural Language Toolkit python library.

```python
from nltk.corpus import stopwords
stop = stopwords.words('english')
data['Lyrics'] = data['Lyrics'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
data['Lyrics'].head()
```

### Remove Frequent Words


If we want to catch common words that might have slipped through the stopword removal, we can build out a list of the most common words remaining in our text.

Here we have built a list of the 10 most common words. Some of these words might actually be relevant to our analysis so it is important to be careful with this method.

```python
freq = pd.Series(' '.join(data['Lyrics']).split()).value_counts()[:10]
freq
```

We now could follow the same procedure with which we removed stopwords to remove the most frequent words but we won't here.

```python
#freq = list(freq.index)
#data['Lyrics'] = data['Lyrics'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
#data['Lyrics'].head()
```

## Stemming


Stemming is the process of removing suffices, like "ed" or "ing".

We will use another standard NLTK package, PorterStemmer, to do the stemming.

```python
from nltk.stem import PorterStemmer
st = PorterStemmer()
data['Lyrics'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
```

As we can see "eyes" became "eye", which could help an analysis, but "castle" became "castl" which is less helpful.


## Lemmatization


Lemmatization is often a more useful approach than stemming because it leverages an understanding of the word itself to convert the word back to its root word. However, this means lemmatization is less aggressive than stemming (probably a good thing).

```python
#import nltk
#nltk.download('wordnet')
```

```python
from textblob import Word
data['Lyrics'] = data['Lyrics'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
data['Lyrics'].head()
```

Notice how we still caught "eyes" to "eye" but left "castle" as is.


## Visualizing Text


Now, we are going to make a word cloud based on our data set. Wordclouds are great to get a sense of your text but are difficult to use for actual analysis work.

```python
import wordcloud
```

```python
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format ='retina'
#stop = set(STOPWORDS)
#stop.update(["mash"])
```

If you want to update the stopwords after you see the wordcloud, type them into the empty list above and remove the # sign.

```python
word = " ".join(data['Lyrics'])
wordcloud = WordCloud(stopwords=stop, background_color="black").generate(word)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
```

The above code creates a wordcloud based off all the words (except for stop words) in all of the stories. While this can be fun, it may not be as interesting as a wordcloud for a single story. Let's compare:

```python
for i in range(0, 11):
    word = data['Lyrics'][i]
    wordcloud = WordCloud(stopwords=stop, background_color="black").generate(word)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
```

# Advanced Text Processing


This section focuses on more complex methods of analyzing textual data. We will continue to work with our same DataFrame.


## N-grams


N-grams are combinations of multiple words as they appear in the text. The N refers to the number of words captured in the list. N-grams with N=1 are referred unigrams and are just a nested list of all the words in the text. Following a similar pattern, bigrams (N=2), trigrams (N=3), etc. can be used.

N-grams allow you to capture the structure of the text which can be very useful. For instance, counting the number of bigrams where "said" was preceded by "he" vs "she" could give you an idea of the gender breakdown of speakers in a text. However, if you make your N-grams too long, you lose the ability to make comparisons.

Another concern, especially in very large data sets, is that the memory storage of N-grams scales with N (bigrams are twice as large as unigrams, for example) and the time to process the N-grams can balloon dramatically as well.

All that being said, we would suggest focusing on bigrams and trigrams as useful analysis tools.

```python
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

lemmatizer = WordNetLemmatizer()
n_grams = TextBlob(" ".join(data['Lyrics'])).ngrams(2)
```

```python
characters=[]
for i in ['wolfman', 'vampire', 'monster']:
     characters.append(lemmatizer.lemmatize(i))
```

```python
for n in n_grams:
    if n[1] in characters:
        print(n)
```

```python
from nltk import ngrams
from collections import Counter

ngram_counts = Counter(ngrams(" ".join(data['Lyrics']).split(), 2))
for n in ngram_counts.most_common(10):
    print(n)
```

## Term Frequency


Term Frequency is a measure of how often a term appears in a document. There are different ways to define this but the simplest is a raw count of the number of times each term appears.

There are other ways of defining this including a true term frequency and a log scaled definition. All three have been implemented below but the default will the raw count definition, as it matches with the remainder of the definitions in this tutorial.

|Definition|Formula|
|---|---|
|Raw Count|$$f_{t,d}$$|
|Term Frequency|$$\frac{f_{t,d}}{\sum_{t'\in d}f_{t',d}}$$|
|Log Scaled|$$\log(1+f_{t,d})$$|



```python
## Raw Count Definition
tf1 = (data['Lyrics']).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()

## Term Frequency Definition
#tf1 = (data['Lyrics']).apply(lambda x: (pd.value_counts(x.split(" ")))/len(x.split(" "))).sum(axis = 0).reset_index() 

## Log Scaled Definition
#tf1 = (data['Lyrics']).apply(lambda x: 1.0+np.log(pd.value_counts(x.split(" ")))).sum(axis = 0).reset_index() 

tf1.columns = ['words','tf']
tf1.sort_values(by='tf', ascending=False)[:10]
```

## Inverse Document Frequency


Inverse Document Frequency is a measure of how common or rare a term is across multiple documents. That gives a measure of how much weight that term carries.

For a more concrete analogy of this, imagine a room full of NBA players; here a 7 foot tall person wouldn't be all that shocking. However if you have a room full of kindergarten students, a 7 foot tall person would be a huge surprise.

The simplest and standard definition of Inverse Document Frequency is to take the logarithm of the ratio of the number of documents containing a term to the total number of documents.

$$-\log\frac{n_t}{N} = \log\frac{N}{n_t}$$


```python
for i,word in enumerate(tf1['words']):
    tf1.loc[i, 'idf'] = np.log(data.shape[0]/(len(data[data['Lyrics'].str.contains(word)])))

tf1.sort_values(by='idf', ascending=False)[:10]
```

## Term Frequency – Inverse Document Frequency ([TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf))


Term Frequency – Inverse Document Frequency (TF-IDF) is a composite measure of both Term Frequency and Inverse Document Frequency.

From [Wikipedia](https://en.wikipedia.org/wiki/Tf%E2%80%93idf):
"A high weight in TF–IDF is reached by a high term frequency (in the given document) and a low document frequency of the term in the whole collection of documents; the weights hence tend to filter out common terms"

More concisely, a high TD-IDF says that a word is very important in the documents in which it appears.

There are a few weighting schemes for TF-IDF. Here we use scheme (1).

|Weighting Scheme|Document Term Weight|
|---|---|
|(1)|$$f_{t,d}\cdot\log\frac{N}{n_t}$$|
|(2)|$$1+\log(f_{t,d})$$|
|(3)|$$(1+\log(f_{t,d}))\cdot\log\frac{N}{n_t}$$|

```python
tf1['tfidf'] = tf1['tf'] * tf1['idf']
tf1.sort_values(by='tfidf', ascending=False)[:10]
```

It is worth noting that the *sklearn* library has the ability to directly calculate a TD-IDF matrix. (We'll use this later)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',
 stop_words= 'english',ngram_range=(1,1))
data_vect = tfidf.fit_transform(data['Lyrics'])

data_vect
```

## Similarity

```python
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
pd.options.mode.chained_assignment = None  # default='warn' #suppress our crossjoin warning message
warnings.filterwarnings("ignore", category=FutureWarning) # This suppresses a warning from scikit learn that they are going to update their code
```

One thing we can look at is how similar two texts are. This has a practical use when looking for plagiarism, but can also be used to compare author's styles. To do this there are a few ways we can measure similarity.


First we need to set up our two sets of texts. Here we have the ability to choose the size of our sets and whether we want the first n texts from our full data set or just a random sample. Keep in mind that if you want to use two dissimilar sets, you won't have a control value (something x itself).

```python
set1 = data[["index",'Lyrics']]
set2 = data[["index",'Lyrics']]
```

```python
#To let us do a "cross join": Every row in one gets matched to all rows in the other
set1['key']=1 
set2['key']=1

similarity = pd.merge(set1, set2, on ='key', suffixes=('_1', '_2')).drop("key", 1)
```

### Jaccard Similarity


The first is using a metric called the [Jaccard Index](https://en.wikipedia.org/wiki/Jaccard_index). This is just taking the intersection of two sets of things (in our case, words or n-grams) and dividing it by the union of those sets. This gives us a metric for understanding how the word usage compares but doesn't account for repeated words since the union and intersections just take unique words. One advantage though is that we can easily extend the single word similarity to compare bi-grams and other n-grams if we want to examine phrase usage.

$$S_{J}(A,B)=\frac{A \cap B}{A \cup B}$$


```python
def jaccard(row, n=1):
    
    old=row['Lyrics_1']
    new=row['Lyrics_2']
    
    old_n_grams = [tuple(el) for el in TextBlob(old).ngrams(n)]
    new_n_grams = [tuple(el) for el in TextBlob(new).ngrams(n)]
        
    union = list(set(old_n_grams) | set(new_n_grams))
    intersection = list(set(old_n_grams) & set(new_n_grams))
    
    lu = len(union)
    li = len(intersection)
    
    return (li/lu,li,lu)
```

```python
for i in [1,2]: # Add values to the list for the n-gram you're interested in
    similarity['Jaccard_Index_for_{}_grams'.format(i)]=similarity.apply(lambda x: jaccard(x,i)[0],axis=1)
```

```python
similarity['Intersection']=similarity.apply(lambda x: jaccard(x)[1],axis=1)
```

```python
similarity['Union']=similarity.apply(lambda x: jaccard(x)[2],axis=1)
```

```python
similarity.head()
```

### Cosine Similarity


The second metric we can use is [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity), however there is a catch here. Cosine similarity requires a vector for each word so we make a choice here to use term frequency. You could choose something else, inverse document frequency or tf-idf would both be good choices. Cosine similarity with a term frequency vector gives us something very similar to the Jaccard Index but accounts for word repetition. This makes it better for tracking word importance between two texts.

$$S_{C}(v_1,v_2)=cos(\theta)=\frac{v_1\cdot v_2}{||v_1||\times||v_2||}$$

```python
def get_cosine_sim(str1,str2): 
    vectors = [t for t in get_vectors([str1,str2])]
    return cosine_similarity(vectors)
    
def get_vectors(slist):
    text = [t for t in slist]
    vectorizer = CountVectorizer(text)
    vectorizer.fit(text)
    return vectorizer.transform(text).toarray()

def cosine(row):
    
    old=row['Lyrics_1']
    new=row['Lyrics_2']
    
    return get_cosine_sim(old,new)[0,1]   
```

```python
similarity['Cosine_Similarity']=similarity.apply(lambda x: cosine(x),axis=1)
```

```python
similarity.head()
```

### Visualize Similarity

```python
#metric = 'Cosine_Similarity'
metric = 'Jaccard_Index_for_1_grams'
```

```python
import numpy as np 
import matplotlib.pyplot as plt

index = list(similarity['index_1'].unique())
columns = list(similarity['index_2'].unique())
df = pd.DataFrame(0, index=index, columns=columns)

for i in df.index:
    sub = similarity[(similarity['index_1']==i)]
    for col in df.columns:
        df.loc[i,col]=sub[sub['index_2']==col][metric].iloc[0]
        
plt.pcolor(df)
plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
plt.show()
```

```python
similarity[(similarity["index_1"]!=similarity["index_2"]) & (similarity[metric]>0.0)]\
[['Lyrics_1','index_1','Lyrics_2','index_2',metric]].sort_values(by=metric,ascending=False)
```

## Sentiment Analysis


Sentiment is a way of measuring the overall positivity or negativity in a given text.

To do this we will use the built in sentiment function in the *TextBlob* package. This function will return the polarity and subjectivity scores for each data chunk.

```python
data['Lyrics'].apply(lambda x: TextBlob(x).sentiment)
```

Focusing on the polarity score, we are able to see the overall sentiment of each data chunk. The closer to 1 the more positive and the closer to -1 the more negative.

```python
data['sentiment'] = data['Lyrics'].apply(lambda x: TextBlob(x).sentiment[0] )
data[['Lyrics','sentiment']].head()
```

```python
data[['Lyrics','sentiment']].sort_values(by='sentiment',ascending=False)
```

```python
plot_1 = data[['sentiment']].plot(kind='bar')
```

```python
plot_2 = data[['sentiment']].sort_values(by='sentiment', ascending=False).plot(kind='bar')
```

## Using TF-IDF and Machine Learning


This is significantly more advanced than the rest of the tutorial. This takes the TF-IDF matrix and applies a [k-means clustering algorithm](https://en.wikipedia.org/wiki/K-means_clustering). This groups the texts into clusters of similar terms from the TF-IDF matrix. This algorithm randomly seeds X "means", the values are then clustered into the nearest mean. The centroid of the values in each cluster then becomes the new mean and the process repeats until a convergence is reached.

```python
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

groups = 2

num_clusters = groups
num_seeds = groups
max_iterations = 300
labels_color_map = {
    0: '#20b2aa', 1: '#ff7373', 2: '#ffe4e1', 3: '#005073', 4: '#4d0404',
    5: '#ccc0ba', 6: '#4700f9', 7: '#f6f900', 8: '#00f91d', 9: '#da8c49'
}
pca_num_components = 2
tsne_num_components = 2

# calculate tf-idf of texts
tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',stop_words= 'english',ngram_range=(1,1))
tf_idf_matrix = tfidf.fit_transform(data['Lyrics'])

# create k-means model with custom config
clustering_model = KMeans(
    n_clusters=num_clusters,
    max_iter=max_iterations,
    precompute_distances="auto",
    n_jobs=-1
)

labels = clustering_model.fit_predict(tf_idf_matrix)
#print(labels)

X = tf_idf_matrix.todense()

# ----------------------------------------------------------------------------------------------------------------------

reduced_data = PCA(n_components=pca_num_components).fit_transform(X)
# print(reduced_data)

import matplotlib.patches as mpatches
legendlist=[mpatches.Patch(color=labels_color_map[key],label=str(key))for key in labels_color_map.keys()][:groups]

fig, ax = plt.subplots()
for index, instance in enumerate(reduced_data):
    #print(instance, index, labels[index])
    pca_comp_1, pca_comp_2 = reduced_data[index]
    color = labels_color_map[labels[index]]
    ax.scatter(pca_comp_1, pca_comp_2, c=color)
plt.legend(handles=legendlist)
plt.show()



# t-SNE plot
#embeddings = TSNE(n_components=tsne_num_components)
#Y = embeddings.fit_transform(X)
#plt.scatter(Y[:, 0], Y[:, 1], cmap=plt.cm.Spectral)
#plt.show()

```

```python
tfidf_test = tf1.sort_values(by='tfidf', ascending=False)[:1000]
```

```python
title_groups = np.transpose([labels,data['Lyrics'],data['index']])
```

Keep in mind that each time you run the algorithm, the randomness in generating the initial means will result in different clusters.

```python
for i in range(len(title_groups)):
    data.loc[i,'Group'] = title_groups[i][0]
```

```python
grp=0
data[data['Group']==grp]['Lyrics']
```

```python
for i in range(groups):
    print("")
    print("#### {} ###".format(i))
    print("")
    for el in title_groups:
        if el[0]==i:
            print("{}".format(el[1]))
```

```python

```
