
# Multiclass Text Classification & Comparison 

LEGAL: This notebook is an adaptation of Susan Li's article "Multi-Class Text Classification Model Comparison and Selection", 
which can be found at
https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568 .
The following notebook has been extended and modified; i.e. added / changed / summarized some explanations, added comments, and extended code as well. 

We have a bunch of stack overflow posts, and we want to figure out where they came from. 
The question is, which algorithm is better to use? 

### Imports 


```python
import re
import logging
import pandas as pd 
import numpy as np 
import gensim 
import nltk 
import matplotlib.pyplot as plt 
%matplotlib inline 
import seaborn as sns
from numpy import random 
from bs4 import BeautifulSoup 
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, confusion_matrix  
```

## Exploring the Data 


```python
# Import the dataset 
PATH = "Datasets\\stack-overflow-data.csv"
df = pd.read_csv(PATH) 
df = df[pd.notnull(df['tags'])] # Exclude null entries

# Show the data 
print(df.head(10))

# Split the data by spaces, and sum the aggregated sum  
print("Number of words: ", df['post'].apply(lambda x: len(x.split(' '))).sum()) # ??? 

```

                                                    post           tags
    0  what is causing this behavior  in our c# datet...             c#
    1  have dynamic html load as if it was in an ifra...        asp.net
    2  how to convert a float value in to min:sec  i ...    objective-c
    3  .net framework 4 redistributable  just wonderi...           .net
    4  trying to calculate and print the mean and its...         python
    5  how to give alias name for my website  i have ...        asp.net
    6  window.open() returns null in angularjs  it wo...      angularjs
    7  identifying server timeout quickly in iphone  ...         iphone
    8  unknown method key  error in rails 2.3.8 unit ...  ruby-on-rails
    9  from the include  how to show and hide the con...      angularjs
    Number of words:  10286120
    

We have 10286120 words in the data! 
Now we will set up the tags (i.e. our targets) . We can also check their distribution. 


```python
my_tags = ['java','html','asp.net','c#','ruby-on-rails',
           'jquery','mysql','php','ios','javascript','python',
           'c','css','android','iphone','sql','objective-c',
           'c++','angularjs','.net']

sns.set()
plt.figure(1, figsize=(10,4))
df.tags.value_counts().plot(kind='bar') 
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1ebb3aea548>




![png](output_6_1.png)


We can see that the classes are pretty well-balanced. 
We also want to have a look at a few possts and tag pairs. 


```python
def print_plot(index): 
    example = df[df.index == index][['post','tags']].values[0]
    if len(example) > 0: 
        print(example[0])
        print('Tag: ', example[1])
        
print_plot(20)
```

    java - hackerrank.com 30 days code day 6 hello stackoverflow peeps! working learn improve coding skills hackerrank.com 30 days code. day 6 issues figuring getting error message: ~ response stdout ~ done searching google within stackoverflow found others error using standard o. feel like missing code: import java.io.* import java.util.* import java.text.* import java.math.* import java.util.regex.* public class solution public static void main string args * enter code here. read input stdin. print output stdout. class named solution. * scanner sc = new scanner system.in int cases = sc.nextint cases > 0 getword cases-- public static void getword save input string scanner sc = new scanner system.in string userinput = sc.nextline convert string character array char inputchararray = userinput.tochararray setup output strings string evenoutputstring = string oddoutputstring = iterate array int = 0 <= userinput.length i++ check index even % 2 == 0 add even output string evenoutputstring = evenoutputstring + inputchararray else add odd output string oddoutputstring = oddoutputstring + inputchararray output final output one line seperated single space system.out.println evenoutputstring + + oddoutputstring challenge found here: https: www.hackerrank.com challenges 30-review-loop problem first input number determines many test cases occur following inputs single words supposed put array sort even odd index slots print one line separated space. example input: 2 hacker rank example output: hce akr rn ak tried moving cases > 0 code block method case args main method part problem got results. also tried .tostring inputchararray output string concatenation. also trying stay away stringbuilder stay within intended scope challenge.
    Tag:  java
    

Notice that the text is pretty raw! We need to clean it up first. 

## Text Preprocessing 

For this particular example, the cleaing step includeds HTML, removing stopwords, change text to lower case, remove punctuation , remove bad characters and so on 


```python
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]') # Values to replace
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]') # Symbols to delete
STOPWORDS = set(stopwords.words('english')) # stopwords to remove
```


```python
def clean_text(text): 
    """
    text: a string
    return : modified initial string 
    """ 
    
    text = BeautifulSoup(text,'lxml').text # HTML decoding 
    text = text.lower() # lowercase 
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # delete given symbols
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # filter stopwords and join 
    
    return text 
```

Now we can apply the cleaning steps to our text 


```python
df['post'] = df['post'].apply(clean_text) 
print_plot(20) 
```

    java - hackerrank.com 30 days code day 6 hello stackoverflow peeps! working learn improve coding skills hackerrank.com 30 days code. day 6 issues figuring getting error message: ~ response stdout ~ done searching google within stackoverflow found others error using standard o. feel like missing code: import java.io.* import java.util.* import java.text.* import java.math.* import java.util.regex.* public class solution public static void main string args * enter code here. read input stdin. print output stdout. class named solution. * scanner sc = new scanner system.in int cases = sc.nextint cases > 0 getword cases-- public static void getword save input string scanner sc = new scanner system.in string userinput = sc.nextline convert string character array char inputchararray = userinput.tochararray setup output strings string evenoutputstring = string oddoutputstring = iterate array int = 0 <= userinput.length i++ check index even % 2 == 0 add even output string evenoutputstring = evenoutputstring + inputchararray else add odd output string oddoutputstring = oddoutputstring + inputchararray output final output one line seperated single space system.out.println evenoutputstring + + oddoutputstring challenge found here: https: www.hackerrank.com challenges 30-review-loop problem first input number determines many test cases occur following inputs single words supposed put array sort even odd index slots print one line separated space. example input: 2 hacker rank example output: hce akr rn ak tried moving cases > 0 code block method case args main method part problem got results. also tried .tostring inputchararray output string concatenation. also trying stay away stringbuilder stay within intended scope challenge.
    Tag:  java
    

Way better! 


```python
df['post'].apply(lambda x: len(x.split(' '))).sum()
```




    3470652



We have 3763604 words instead! (about ~3 million) 

## Data Splitting 

We will now split the data into train and test sets using sklearn


```python
X = df.post
y = df.tags 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, 
                                                   random_state= 42)
# Observe shapes match 
print(X.shape)
print(y.shape)

target_names = list(set(y)) 
print("target names: ", target_names)
```

    (40000,)
    (40000,)
    target names:  ['sql', '.net', 'c++', 'ios', 'mysql', 'java', 'c', 'android', 'javascript', 'angularjs', 'html', 'iphone', 'css', 'jquery', 'objective-c', 'php', 'ruby-on-rails', 'c#', 'asp.net', 'python']
    

## Classifiers  
Now that we have our features, we will perform **feature engineering** 
- 1. We will convert out text documents to a matrix of token counts (`CountVectorizer`)   
- 2. We will then transform a the count matrix to a normalized tf-idf representation. (`TfidfTransformer`) 
- 3.Finally, we will train several classifiers from Scikit-Learn. 

### Pipeline 

We will employ a pipeline in the following way

`vectorizer` -> `tfidf-transformer` -> `normalizer` -> `classifier` 

Where: 
 - **Vectorizer** : transforms the input text into one hot-encoded vectors 
 - **Transformer** : assigns a tfidf based weight to the vectors 
 - **Normalizer** : normalizes the vectors so they have the same range 
 - **Classifier** : the classifier algorithm


```python
# Imports
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV 
from scipy.stats import randint as randint 
from scipy.stats import uniform 
```

### Multinomial Naive Bayes 


```python
from sklearn.naive_bayes import MultinomialNB 

# Create the pipleine 
nb = Pipeline([('vect', CountVectorizer()),  # Vectorizer
               ('tfidf', TfidfTransformer()), # Transformer
               ('norm', Normalizer()), # Normalize
               ('clf', MultinomialNB()),  # Classifier 
              ])

# Fit the dataset
nb.fit(X_train, y_train) 

# Get predictions
y_pred = nb.predict(X_test) 

# Get the accuracy classification report 
MultinomialNB_acc = accuracy_score(y_pred, y_test) 
print("Multinomial NB accuracy %s" % MultinomialNB_acc)
print(classification_report(y_test, y_pred, target_names=my_tags))
```

    Multinomial NB accuracy 0.7415833333333334
                   precision    recall  f1-score   support
    
             java       0.66      0.63      0.64       613
             html       0.94      0.87      0.91       620
          asp.net       0.83      0.91      0.87       587
               c#       0.68      0.81      0.74       586
    ruby-on-rails       0.75      0.86      0.80       599
           jquery       0.71      0.54      0.61       589
            mysql       0.79      0.75      0.77       594
              php       0.67      0.91      0.77       610
              ios       0.65      0.58      0.61       617
       javascript       0.59      0.65      0.61       587
           python       0.72      0.51      0.60       611
                c       0.83      0.80      0.81       594
              css       0.81      0.60      0.69       619
          android       0.64      0.83      0.73       574
           iphone       0.61      0.82      0.70       584
              sql       0.69      0.65      0.67       578
      objective-c       0.80      0.78      0.79       591
              c++       0.90      0.82      0.86       608
        angularjs       0.93      0.89      0.91       638
             .net       0.74      0.63      0.68       601
    
         accuracy                           0.74     12000
        macro avg       0.75      0.74      0.74     12000
     weighted avg       0.75      0.74      0.74     12000
    
    

Now we would like to find the best parameter uing a **Randomized Search Cross-Validation** (`RandomizedSearchCV`) search to see if we can improve our model's accuracy


```python
params = {"vect__ngram_range": [(1,1),(1,2),(2,2)], 
            "tfidf__use_idf":[True, False], 
            "clf__alpha":uniform(1e-2, 1e-3)} 

seed= 551 

random_search = RandomizedSearchCV(nb, 
                                  param_distributions = params, 
                                  cv=5, verbose=10, 
                                  random_state=seed, 
                                  n_iter=1) 

random_search.fit(X_train, y_train) 

## CV Results and final evaluation ## 

report = random_search.cv_results_ 
y_pred = random_search.predict(X_test) 
CV_report = classification_report(y_test, y_pred, 
                                 target_names=target_names) 

print("CV report: \n", CV_report)
print("Report: \n", report)
```

    Fitting 5 folds for each of 1 candidates, totalling 5 fits
    

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    

    [CV] clf__alpha=0.010640064047894703, tfidf__use_idf=False, vect__ngram_range=(1, 1) 
    [CV]  clf__alpha=0.010640064047894703, tfidf__use_idf=False, vect__ngram_range=(1, 1), score=0.734, total=   4.8s
    [CV] clf__alpha=0.010640064047894703, tfidf__use_idf=False, vect__ngram_range=(1, 1) 
    

    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    4.7s remaining:    0.0s
    

    [CV]  clf__alpha=0.010640064047894703, tfidf__use_idf=False, vect__ngram_range=(1, 1), score=0.726, total=   4.4s
    [CV] clf__alpha=0.010640064047894703, tfidf__use_idf=False, vect__ngram_range=(1, 1) 
    

    [Parallel(n_jobs=1)]: Done   2 out of   2 | elapsed:    9.2s remaining:    0.0s
    

    [CV]  clf__alpha=0.010640064047894703, tfidf__use_idf=False, vect__ngram_range=(1, 1), score=0.724, total=   4.4s
    [CV] clf__alpha=0.010640064047894703, tfidf__use_idf=False, vect__ngram_range=(1, 1) 
    

    [Parallel(n_jobs=1)]: Done   3 out of   3 | elapsed:   13.6s remaining:    0.0s
    

    [CV]  clf__alpha=0.010640064047894703, tfidf__use_idf=False, vect__ngram_range=(1, 1), score=0.741, total=   4.8s
    [CV] clf__alpha=0.010640064047894703, tfidf__use_idf=False, vect__ngram_range=(1, 1) 
    

    [Parallel(n_jobs=1)]: Done   4 out of   4 | elapsed:   18.5s remaining:    0.0s
    

    [CV]  clf__alpha=0.010640064047894703, tfidf__use_idf=False, vect__ngram_range=(1, 1), score=0.737, total=   4.5s
    

    [Parallel(n_jobs=1)]: Done   5 out of   5 | elapsed:   23.0s remaining:    0.0s
    [Parallel(n_jobs=1)]: Done   5 out of   5 | elapsed:   23.0s finished
    

    CV report: 
                    precision    recall  f1-score   support
    
              sql       0.58      0.67      0.62       613
             .net       0.94      0.85      0.90       620
              c++       0.89      0.91      0.90       587
              ios       0.72      0.74      0.73       586
            mysql       0.78      0.81      0.80       599
             java       0.69      0.57      0.62       589
                c       0.73      0.77      0.75       594
          android       0.74      0.88      0.80       610
       javascript       0.62      0.65      0.64       617
        angularjs       0.57      0.55      0.56       587
             html       0.60      0.58      0.59       611
           iphone       0.84      0.74      0.79       594
              css       0.73      0.64      0.68       619
           jquery       0.72      0.77      0.74       574
      objective-c       0.66      0.74      0.70       584
              php       0.67      0.65      0.66       578
    ruby-on-rails       0.80      0.75      0.77       591
               c#       0.85      0.85      0.85       608
          asp.net       0.93      0.91      0.92       638
           python       0.70      0.70      0.70       601
    
         accuracy                           0.74     12000
        macro avg       0.74      0.74      0.74     12000
     weighted avg       0.74      0.74      0.74     12000
    
    Report: 
     {'mean_fit_time': array([3.85491395]), 'std_fit_time': array([0.18855075]), 'mean_score_time': array([0.74366994]), 'std_score_time': array([0.09248282]), 'param_clf__alpha': masked_array(data=[0.010640064047894703],
                 mask=[False],
           fill_value='?',
                dtype=object), 'param_tfidf__use_idf': masked_array(data=[False],
                 mask=[False],
           fill_value='?',
                dtype=object), 'param_vect__ngram_range': masked_array(data=[(1, 1)],
                 mask=[False],
           fill_value='?',
                dtype=object), 'params': [{'clf__alpha': 0.010640064047894703, 'tfidf__use_idf': False, 'vect__ngram_range': (1, 1)}], 'split0_test_score': array([0.7342246]), 'split1_test_score': array([0.7262181]), 'split2_test_score': array([0.72370066]), 'split3_test_score': array([0.74106505]), 'split4_test_score': array([0.73676681]), 'mean_test_score': array([0.73239286]), 'std_test_score': array([0.00650088]), 'rank_test_score': array([1])}
    

### Linear Support Vector Machines 


```python
from sklearn.linear_model import SGDClassifier 

# Create pipeline 
sgd = Pipeline([('vect', CountVectorizer()), 
                ('tfidf', TfidfTransformer()), 
                ('clf', SGDClassifier(loss='hinge', 
                                      penalty='l2', 
                                      alpha=1e-3, 
                                      random_state=42, 
                                      max_iter=5, 
                                      tol=None))
               ])

# fit the model 
sgd.fit(X_train, y_train) 

# get predictions 
y_pred = sgd.predict(X_test) 

SGD_acc = accuracy_score(y_pred, y_test) 
print('Linear SVM accuracy:  %s' % SGD_acc)
print(classification_report(y_test, y_pred,target_names=my_tags))
```

    Linear SVM accuracy:  0.7954166666666667
                   precision    recall  f1-score   support
    
             java       0.78      0.66      0.72       613
             html       0.86      0.93      0.89       620
          asp.net       0.88      0.96      0.92       587
               c#       0.81      0.85      0.83       586
    ruby-on-rails       0.74      0.89      0.81       599
           jquery       0.78      0.43      0.55       589
            mysql       0.84      0.71      0.77       594
              php       0.70      0.95      0.80       610
              ios       0.82      0.56      0.67       617
       javascript       0.73      0.59      0.65       587
           python       0.72      0.65      0.69       611
                c       0.81      0.88      0.84       594
              css       0.79      0.78      0.78       619
          android       0.83      0.86      0.84       574
           iphone       0.83      0.81      0.82       584
              sql       0.71      0.69      0.70       578
      objective-c       0.80      0.92      0.85       591
              c++       0.84      0.95      0.89       608
        angularjs       0.88      0.94      0.91       638
             .net       0.78      0.88      0.83       601
    
         accuracy                           0.80     12000
        macro avg       0.80      0.79      0.79     12000
     weighted avg       0.80      0.80      0.79     12000
    
    

We see a 5% improvement over Bayes! 

### Word2Vec and Logistic Regression 

Now, let's try more complex features than simply counting. For this, we will use the word2vec algorithm , using the Google pretrained word vectors. 


```python
from gensim.models import Word2Vec 

# initilaize the vectors (takes some time)
PATH = "../../Datasets/GoogleNews-vectors-negative300.bin"
wv = gensim.models.KeyedVectors.load_word2vec_format(PATH, binary=True)
wv.init_sims(replace=True) # normalize vectors 
```

We may want to explore some vocabularies 


```python
from itertools import islice 

list(islice(wv.vocab, 13030, 13050))
```




    ['Memorial_Hospital',
     'Seniors',
     'memorandum',
     'elephant',
     'Trump',
     'Census',
     'pilgrims',
     'De',
     'Dogs',
     '###-####_ext',
     'chaotic',
     'forgive',
     'scholar',
     'Lottery',
     'decreasing',
     'Supervisor',
     'fundamentally',
     'Fitness',
     'abundance',
     'Hold']



BOW based approaches includes averaging , summation, weighting addition. 
The common way is to average thw two word vectors. Therefore, we will follow the standard. 


```python
def word_averaging(wv, words): 
    """ 
    Averages a set of words by using the word vectors wv
    
    params: 
        @ wv: Gensim fromatted wordvectors
        @ words: a list of words 
    """
 # initialize
    all_words, mean = set(), []
    
    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.vocab:
            mean.append(wv.vectors_norm[wv.vocab[word].index])
            all_words.add(wv.vocab[word].index)

    if not mean:
        logging.warning("cannot compute similarity with no input %s", words)
        # FIXME: remove these examples in pre-processing
        return np.zeros(wv.vector_size,)

    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)

    return mean  


def word_averaging_list(wv, text_list): 
    """ 
    Averages a text list using the word_averaging function
    """
    return np.vstack([word_averaging(wv, post) for post in text_list])

```

We will tokenize the text and apply the tokenization to the "post" column and apply word vector averaging to tokenized text. 


```python
def w2v_tokenize_text(text): 
    """
    Tokenizes input text by sentences and then by words using nltk
    tokenizers (English)
    """
    tokens = []
    for sent in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(sent, language='english'):
            if len(word) < 2:
                continue
            tokens.append(word)
    
    return tokens

# split into train and test sets 7:3
train, test = train_test_split(df, test_size=0.3, random_state = 42)

# Tokenize the splitted sets 
test_tokenized = test.apply(lambda r: w2v_tokenize_text(r['post']), axis=1).values
train_tokenized = train.apply(lambda r: w2v_tokenize_text(r['post']), axis=1).values

# Average the results  
X_train_word_average = word_averaging_list(wv,train_tokenized)
X_test_word_average = word_averaging_list(wv,test_tokenized) 

# Display some examples 
print(train_tokenized[0:10])
print(X_train_word_average[0:10]) 
```

    WARNING:root:cannot compute similarity with no input ['i=1']
    WARNING:root:cannot compute similarity with no input []
    

    [[ 0.0253806  -0.01552145 -0.00779094 ... -0.06064709 -0.04819088
       0.00663165]
     [ 0.0750726  -0.08307417  0.01276819 ... -0.11526021 -0.02776728
       0.00594368]
     [ 0.03748594  0.03292246 -0.04062274 ... -0.02492273 -0.07283854
       0.02438511]
     ...
     [ 0.02056828  0.01150424 -0.00473571 ... -0.08026548 -0.06164216
       0.03034527]
     [ 0.03031846 -0.03152335  0.0187664  ... -0.02194222 -0.0477215
       0.04721637]
     [ 0.08866245 -0.02157295 -0.03285461 ... -0.05000888  0.01348568
      -0.01409832]]
    

It's time to see how logistic regression classifiers performs on these word-averaging document features! 


```python
from sklearn.linear_model import LogisticRegression 

logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg = logreg.fit(X_train_word_average, train['tags'])
y_pred = logreg.predict(X_test_word_average)
print('accuracy %s' % accuracy_score(y_pred, test.tags))
print(classification_report(test.tags, y_pred,target_names=my_tags))

```

    C:\Users\jairp\AppData\Roaming\Python\Python37\site-packages\sklearn\linear_model\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    C:\Users\jairp\AppData\Roaming\Python\Python37\site-packages\sklearn\linear_model\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    

    accuracy 0.647
                   precision    recall  f1-score   support
    
             java       0.60      0.56      0.58       613
             html       0.74      0.76      0.75       620
          asp.net       0.63      0.64      0.63       587
               c#       0.64      0.64      0.64       586
    ruby-on-rails       0.71      0.78      0.74       599
           jquery       0.44      0.37      0.40       589
            mysql       0.67      0.64      0.66       594
              php       0.74      0.84      0.79       610
              ios       0.61      0.60      0.60       617
       javascript       0.56      0.52      0.54       587
           python       0.56      0.51      0.54       611
                c       0.62      0.62      0.62       594
              css       0.64      0.63      0.64       619
          android       0.58      0.54      0.56       574
           iphone       0.71      0.73      0.72       584
              sql       0.44      0.45      0.44       578
      objective-c       0.68      0.74      0.71       591
              c++       0.77      0.79      0.78       608
        angularjs       0.81      0.82      0.82       638
             .net       0.68      0.72      0.70       601
    
         accuracy                           0.65     12000
        macro avg       0.64      0.65      0.64     12000
     weighted avg       0.64      0.65      0.64     12000
    
    

That was deceiving, the worst so far. 

### Doc2Vec and Logistic Regression 

We can extend the previous ideas also to sentences and documents. 
First, we label the sentences. Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it, and we will do so by using the TaggedDocument method. 


```python
import re
import gensim 
from tqdm import tqdm # will create a cute progress bar
tqdm.pandas(desc='progress-bar')
from gensim.models import Doc2Vec 
from gensim.models.doc2vec import TaggedDocument
from sklearn import utils 

def label_sentences(corpus, label_type): 
    """
    Gensim's Doc2Vec implementation requires each document/paragraph 
    to have a label associated with it.
    We do this by using the TaggedDocument method. The format will be 
    "TRAIN_i" or "TEST_i" where "i" is a dummy index of the post. 
    """
    labeled = [] 
    for i, v in enumerate(corpus): 
        label = label_type + '_' + str(i) 
        labeled.append(TaggedDocument(v.split(), [label])) 
        
    return labeled

# split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df.post, df.tags,
                                                   random_state=0, 
                                                   test_size=0.3) 
# label the sentences
X_train = label_sentences(X_train, 'Train') 
X_test = label_sentences(X_test, 'Test') 
all_data = X_train + X_test # merge both 
```

According to the Gensim doc2vec tutorial, its doc2vec was trained on the entire data and we will do the same. 
Let's have a look a what the tagged documents look like: 


```python
all_data[:2]
```




    [TaggedDocument(words=['full-text', 'search', 'php', 'pdo', 'returning', 'result', 'searched', 'lot', 'matter', 'find', 'wrong', 'setup.', 'trying', 'full-text', 'search', 'using', 'pdo', 'php', 'get', 'results', 'error', 'messages', 'all.', 'table', 'contains', 'customer', 'details:', 'id', 'int', '11', 'auto_increment', 'name', 'varchar', '150', 'lastname', 'varchar', '150', 'company', 'varchar', '250', 'adress', 'varchar', '150', 'postcode', 'int', '5', 'city', 'varchar', '150', 'email', 'varchar', '250', 'phone', 'varchar', '20', 'orgnr', 'varchar', '15', 'timestamp', 'timestamp', 'current_timestamp', 'run', 'sql-query:', 'alter', 'table', 'system_customer', 'add', 'fulltext', 'name', 'lastname', '...', '...', 'except', 'columns', 'id', 'postcode', 'timestamp', '.', 'signs', 'trouble', 'far.', 'idea', 'problem', 'lies', 'db', 'configuration', 'php', 'code', 'goes', 'php:', '$sth', '=', '$dbh->prepare', 'select', 'name', 'lastname', 'company', 'adress', 'city', 'phone', 'email', 'orgnr', '.$db_pre.', 'customer', 'match', 'name', 'lastname', 'company', 'adress', 'city', 'phone', 'email', 'orgnr', ':search', 'boolean', 'mode', 'bind', 'placeholders', '$sth->bindparam', ':search', '$data', '$sth->execute', '$rows', '=', '$sth->fetchall', 'testing', 'print_r', '$dbh->errorinfo', 'empty', '$rows', 'echo', '.....', 'else', 'echo', '....', 'foreach', '$rows', '$row', 'echo', 'echo', '.$row', 'name', '.', '<', 'td>', 'echo', '.$row', 'lastname', '.', '<', 'td>', 'echo', '.$row', 'company', '.', '<', 'td>', 'echo', '.$row', 'phone', '.', '<', 'td>', 'echo', '.$row', 'email', '.', '<', 'td>', 'echo', '.date', 'y-m-d', 'strtotime', '$row', 'timestamp', '.', '<', 'td>', 'echo', '<', 'tr>', 'echo', '....', 'tried', 'change', 'parameter', 'searchquery', 'string', 'like', 'testcompany', 'somename', 'boolean', 'mode', 'also', 'read', 'word', 'found', '50%', 'rows', 'counts', 'common', 'word.', 'pretty', 'sure', 'case', 'uses', 'specific', 'words', 'table', 'uses', 'myisam', 'engine', 'get', 'results', 'error', 'messages.', 'please', 'help', 'point', 'wrong', 'thank'], tags=['Train_0']),
     TaggedDocument(words=['select', 'everything', '1', 'table', 'x', 'rows', 'another', 'im', 'making', 'join', 'query', 'like:', 'select', '*', 'clothes', 'c', 'join', 'style', 'c.styleid', '=', 's.sylelid', 'clothesid', '=', '19', 'dont', 'want', 'select', 'everything', 'style', 'want', 'select', 'everything', 'clothes', '20', 'rows', 'select', '1', 'row', '10', 'style', 'easyest', 'way', 'without', 'select', 'every', 'row', 'clothes', '20', 'things', 'select', 'like:', 'select', 'c.id', 'c.description', 'c.name', 'c.size', 'c.brand', 's.name', 'clothes', 'c', 'join', 'style', 'c.styleid', '=', 'st.sylelid', 'clothesid', '=', '19', 'would', 'fastest', 'way', 'possibillity'], tags=['Train_1'])]



When trianing, we will vary the following parameters: 
- `dm=0` : distributed bag of words (DBOW) is used 
- `vector_size=300` : 300 dimensional vector dimensional feature vectors 
- `negative=5` : specifies how many "noise_words" should be drawn  
- `min_count=1` : ignores all words with total frequency lower than this
- `alpha= 0.065` : the initial learning rae 

We initalize the model and train for 30 epochs


```python
# Instantiate the model with the given parameters
model_dbow = Doc2Vec(dm=0, vector_size=300, 
                     negative=5, min_count=1,
                     min_alpha=0.065) 

# build vocabulary from data and wrap all data in tqdm to track progress
model_dbow.build_vocab([x for x in tqdm(all_data)])

# train the model for 30 epochs: 
for epoch in range(30):
    print("Epoch ", epoch)
    model_dbow.train(utils.shuffle([x for x in tqdm(all_data)]), total_examples=len(all_data), epochs=1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha

```

    
      0%|          | 0/40000 [00:00<?, ?it/s]
    100%|██████████| 40000/40000 [00:00<00:00, 547316.33it/s]
      0%|          | 0/40000 [00:00<?, ?it/s]
    100%|██████████| 40000/40000 [00:00<00:00, 576479.18it/s]
      0%|          | 0/40000 [00:00<?, ?it/s]
    100%|██████████| 40000/40000 [00:00<00:00, 937599.39it/s]
      0%|          | 0/40000 [00:00<?, ?it/s]
    100%|██████████| 40000/40000 [00:00<00:00, 782501.15it/s]
      0%|          | 0/40000 [00:00<?, ?it/s]
    100%|██████████| 40000/40000 [00:00<00:00, 948503.00it/s]
      0%|          | 0/40000 [00:00<?, ?it/s]
    100%|██████████| 40000/40000 [00:00<00:00, 807505.37it/s]
      0%|          | 0/40000 [00:00<?, ?it/s]
    100%|██████████| 40000/40000 [00:00<00:00, 866005.75it/s]
      0%|          | 0/40000 [00:00<?, ?it/s]
    100%|██████████| 40000/40000 [00:00<00:00, 862262.61it/s]
      0%|          | 0/40000 [00:00<?, ?it/s]
    100%|██████████| 40000/40000 [00:00<00:00, 722996.93it/s]
      0%|          | 0/40000 [00:00<?, ?it/s]
    100%|██████████| 40000/40000 [00:00<00:00, 631225.49it/s]
      0%|          | 0/40000 [00:00<?, ?it/s]
    100%|██████████| 40000/40000 [00:00<00:00, 827384.97it/s]
      0%|          | 0/40000 [00:00<?, ?it/s]
    100%|██████████| 40000/40000 [00:00<00:00, 1138704.46it/s]
      0%|          | 0/40000 [00:00<?, ?it/s]
    100%|██████████| 40000/40000 [00:00<00:00, 959290.08it/s]
      0%|          | 0/40000 [00:00<?, ?it/s]
    100%|██████████| 40000/40000 [00:00<00:00, 647366.54it/s]
      0%|          | 0/40000 [00:00<?, ?it/s]
    100%|██████████| 40000/40000 [00:00<00:00, 986970.53it/s]
      0%|          | 0/40000 [00:00<?, ?it/s]
    100%|██████████| 40000/40000 [00:00<00:00, 854829.01it/s]
      0%|          | 0/40000 [00:00<?, ?it/s]
    100%|██████████| 40000/40000 [00:00<00:00, 733346.85it/s]
      0%|          | 0/40000 [00:00<?, ?it/s]
    100%|██████████| 40000/40000 [00:00<00:00, 548033.25it/s]
      0%|          | 0/40000 [00:00<?, ?it/s]
    100%|██████████| 40000/40000 [00:00<00:00, 989304.31it/s]
      0%|          | 0/40000 [00:00<?, ?it/s]
    100%|██████████| 40000/40000 [00:00<00:00, 650849.81it/s]
      0%|          | 0/40000 [00:00<?, ?it/s]
    100%|██████████| 40000/40000 [00:00<00:00, 704498.79it/s]
      0%|          | 0/40000 [00:00<?, ?it/s]
    100%|██████████| 40000/40000 [00:00<00:00, 649786.05it/s]
      0%|          | 0/40000 [00:00<?, ?it/s]
    100%|██████████| 40000/40000 [00:00<00:00, 507112.73it/s]
      0%|          | 0/40000 [00:00<?, ?it/s]
    100%|██████████| 40000/40000 [00:00<00:00, 770427.57it/s]
      0%|          | 0/40000 [00:00<?, ?it/s]
    100%|██████████| 40000/40000 [00:00<00:00, 672231.43it/s]
      0%|          | 0/40000 [00:00<?, ?it/s]
    100%|██████████| 40000/40000 [00:00<00:00, 1336616.95it/s]
      0%|          | 0/40000 [00:00<?, ?it/s]
    100%|██████████| 40000/40000 [00:00<00:00, 882305.52it/s]
      0%|          | 0/40000 [00:00<?, ?it/s]
    100%|██████████| 40000/40000 [00:00<00:00, 790501.85it/s]
      0%|          | 0/40000 [00:00<?, ?it/s]
    100%|██████████| 40000/40000 [00:00<00:00, 403534.18it/s]
      0%|          | 0/40000 [00:00<?, ?it/s]
    100%|██████████| 40000/40000 [00:00<00:00, 835660.22it/s]
      0%|          | 0/40000 [00:00<?, ?it/s]
    100%|██████████| 40000/40000 [00:00<00:00, 519285.01it/s]

Next, we get the vectors from the trained doc2vec model


```python
def get_vectors(model, corpus_size, vectors_size, vectors_type): 
    """ 
    Gets the vectors from trained doc2vec model 
    params: 
        @ model: Trained Doc2vec model 
        @ corpus_size: Size of the data 
        @ vectors_size: Size (dimension) of the embedding vectors 
        @ vectors_type: Training or Testing vectors
        
        @ return: list of vectors 
    """
    # create zero matrix with zeros
    vectors = np.zeros((corpus_size, vectors_size)) 
    
    for i in range(0, corpus_size): 
        # prefix is trian or test, attach a number 
        prefix = vectors_type + '_' + str(i) 
        # obtain the vector by looking it up in the model docvecs
        vectors[i] = model.docvecs[prefix] 
        
    return vectors

# Obtain the trian and test vectors
train_vectors_dbow = get_vectors(model=model_dbow, 
                                 corpus_size=len(X_train), 
                                 vectors_size=300, 
                                 vectors_type='Train') 
test_vectors_dbow = get_vectors(model=model_dbow, 
                                corpus_size=len(X_test), 
                                vectors_size=300, 
                                vectors_type='Test') 

# Display some vectors
print(train_vectors_dbow[0:5])
```

    [[ 0.51379532 -0.16376209  0.20214991 ...  0.0720263  -0.34361973
       0.41452244]
     [ 0.19558232 -0.25855532  0.20807463 ...  0.39630252 -0.35689777
       0.33900124]
     [ 0.13342437 -0.27511063  0.32747039 ...  0.54672712 -0.12382772
       0.90136272]
     [ 0.09709211 -0.18170355  0.12497407 ...  0.25342241 -0.32970548
      -0.01656674]
     [ 0.35581207 -0.40807804  0.28488603 ...  0.10464772 -0.14527442
       0.18582176]]
    

Finallly, we get a logistic regression model trained by the doc2vec features


```python
logreg = LogisticRegression(n_jobs=1, C=1e5)  
logreg = logreg.fit(train_vectors_dbow, y_train)
y_pred = logreg.predict(test_vectors_dbow) 
print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=my_tags))

```

    C:\Users\jairp\AppData\Roaming\Python\Python37\site-packages\sklearn\linear_model\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    C:\Users\jairp\AppData\Roaming\Python\Python37\site-packages\sklearn\linear_model\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    

    accuracy 0.7458333333333333
                   precision    recall  f1-score   support
    
             java       0.65      0.61      0.63       589
             html       0.82      0.83      0.82       661
          asp.net       0.86      0.85      0.86       606
               c#       0.70      0.69      0.70       613
    ruby-on-rails       0.79      0.83      0.81       601
           jquery       0.65      0.63      0.64       585
            mysql       0.82      0.75      0.78       621
              php       0.78      0.79      0.78       587
              ios       0.63      0.64      0.64       560
       javascript       0.61      0.55      0.58       611
           python       0.61      0.61      0.61       593
                c       0.75      0.78      0.77       581
              css       0.77      0.72      0.75       608
          android       0.81      0.83      0.82       593
           iphone       0.75      0.71      0.73       592
              sql       0.64      0.62      0.63       597
      objective-c       0.79      0.83      0.81       604
              c++       0.86      0.91      0.88       610
        angularjs       0.89      0.93      0.91       595
             .net       0.71      0.78      0.75       593
    
         accuracy                           0.75     12000
        macro avg       0.74      0.74      0.74     12000
     weighted avg       0.74      0.75      0.74     12000
    
    

We achieve an accuracy of 80% , which is 1% higher than SVM. 

### BOW with Keras 

Finally, we will use Keras. The algorithm goes as follows: 
- Separate the data into training and test sets. 
- Use tokenizer methods to count the unique words in out vocabulary and assign each of those words to indices. 
- Calling `fit_on_text()` automatically creates a word index lookup for our vocabulary. 
- We limit our vocabulary to the top words by passing a `num_words` parameter to the tokenizer
- With out tokenizer, we can now use the `text_to_matrix` method to create the training data we will pass to our model. 
- After we transform our features and labels in a format Keras can read, we are ready to build our text classification model. 
- When we build out model, all we need to do is to tell Keras the shape of our input data, output data, and the type of each later. 
- When training the model, we'll call the fit() method, pass it out training data and labels, batch size and epochs


```python
import os 
import itertools 

import numpy as np 
import pandas as pd 
import tensorflow as tf
import matplotlib.pyplot as plt 

from sklearn.preprocessing import LabelBinarizer, LabelEncoder 
from sklearn.metrics import confusion_matrix 

from tensorflow import keras 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Activation, Dropout 
from tensorflow.keras.preprocessing import text, sequence 
from tensorflow.keras import utils 
```


```python
# 1. Split data into train and test sets
train_size = int(len(df)* .7 ) # 70% 

train_posts = df['post'][:train_size] # First 70%  of data
train_tags = df['tags'][:train_size] 

test_posts = df['post'][train_size:] # Last 30% of data 
test_tags = df['tags'][train_size:] 

# 2. Tokenizer Methods
max_words = 1000
tokenize = text.Tokenizer(num_words=max_words, char_level=False) 
tokenize.fit_on_texts(train_posts) # only fit train set  

# 3. Convert train and test sets to matrix  
X_train = tokenize.texts_to_matrix(train_posts)
X_test = tokenize.texts_to_matrix(test_posts) 

# 4. Encode into a format Keras can understand 

# 4.1 Encode the labels 
encoder = LabelEncoder() 
encoder.fit(train_tags) # only fit train tags 
y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags) 

# 4.2 Convert to categorical 
num_classes = np.max(y_train) + 1 
y_train = utils.to_categorical(y_train, num_classes) 
y_test = utils.to_categorical(y_test, num_classes) 

# 5. Build the model 

model =Sequential() 
model.add(Dense(512, input_shape=(max_words,))) # 512 neurons
model.add(Activation('relu'))
model.add(Dropout(0.5)) 
model.add(Dense(num_classes)) # Output layer 
model.add(Activation('softmax')) 

# 6. Compile the model 
model.compile(loss='categorical_crossentropy', # since we're dealing with classes
              optimizer='adam', 
             metrics=['accuracy']) 

# 7. Train the model 
history = model.fit(X_train, y_train, 
                    batch_size=32, 
                    epochs=3, 
                    verbose=1, 
                    validation_split=0.1) 
```

    Train on 25200 samples, validate on 2800 samples
    Epoch 1/3
    25200/25200 [==============================] - 9s 353us/sample - loss: 0.9880 - acc: 0.7288 - val_loss: 0.6306 - val_acc: 0.8107
    Epoch 2/3
    25200/25200 [==============================] - 5s 214us/sample - loss: 0.5447 - acc: 0.8283 - val_loss: 0.6187 - val_acc: 0.8121
    Epoch 3/3
    25200/25200 [==============================] - 5s 212us/sample - loss: 0.4393 - acc: 0.8560 - val_loss: 0.6341 - val_acc: 0.8021
    

From this we obtain the accuracy: 


```python
score = model.evaluate(X_test, y_test,
                       batch_size=32, verbose=1)
print('Test accuracy:', score[1])
```

    12000/12000 [==============================] - 2s 130us/sample - loss: 0.6207 - acc: 0.8033
    Test accuracy: 0.80325
    

Now we can predict the labels. 


```python
y_pred = model.predict(X_test)

tags = list(set(train_tags))
print("Tags list: \n\n", tags)
tags1 = encoder.transform(tags)
print("\nEncoded tags tags: \n\n", tags1)
tags2 = utils.to_categorical(tags1, num_classes)
print("\nCategorical tags: \n\n", tags2, "\n")

# Create a mapping from tags to caegorical 
tags_to_cat = dict(zip(tags, tags1))
print("\nTags to categorical: \n\n", tags_to_cat)

pred = list(y_pred[1000])
print(pred.index(max(pred)))

# Lengths
print(len(X_test)) 
print(len(test_posts)) 

print(list(test_posts)[1000])
```

    Tags list: 
    
     ['iphone', 'jquery', 'angularjs', 'ruby-on-rails', 'mysql', 'objective-c', 'php', 'c++', 'android', 'ios', 'asp.net', 'c', 'css', 'javascript', '.net', 'html', 'sql', 'c#', 'python', 'java']
    
    Encoded tags tags: 
    
     [10 13  2 18 14 15 16  6  1  9  3  4  7 12  0  8 19  5 17 11]
    
    Categorical tags: 
    
     [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
     [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
     [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]] 
    
    
    Tags to categorical: 
    
     {'iphone': 10, 'jquery': 13, 'angularjs': 2, 'ruby-on-rails': 18, 'mysql': 14, 'objective-c': 15, 'php': 16, 'c++': 6, 'android': 1, 'ios': 9, 'asp.net': 3, 'c': 4, 'css': 7, 'javascript': 12, '.net': 0, 'html': 8, 'sql': 19, 'c#': 5, 'python': 17, 'java': 11}
    11
    12000
    12000
    java algorithim work time using distance time looking implementation java using method for: time = distance speed time minutes distance metres speed kilometres per hour
    


```python
def get_predictions(y_preds): 
    """ 
    Obtains predictions for an input list of lists 
    """
    
    tags = list(set(train_tags)) # Obtain list of classes
    tags_categorical = encoder.transform(tags) # obtain encoded  
    
    # Create mapping
    encoded_to_tag = dict(zip(tags_categorical, tags)) 
    
    preds = [list(pred) for pred in y_preds] # convert format to list 
    preds = [pred.index(max(pred)) for pred in preds] # Get the prediction 
    preds = [encoded_to_tag[i] for i in preds] # map to tag 
    
    return preds

# Example
print("Example texts: \n")
texts = list(test_posts[1000:1004]) 
for i, text in enumerate(texts): 
    print("Text ", i+1, ":\n", text)
    
    
real_tags = list(test_tags[1000:1004]) # Get real tags 
pred_tags = model.predict(X_test[1000:1004]) # Predict with the model 
pred_tags = get_predictions(pred_tags) # Get prediction tags

print("\nReal tags: " ,real_tags) 
print("Predicted tags: ", pred_tags)
```

    Example texts: 
    
    Text  1 :
     java algorithim work time using distance time looking implementation java using method for: time = distance speed time minutes distance metres speed kilometres per hour
    Text  2 :
     android: create navigation bar seen apps slightly translucent black bar bottom screen buttons menu bar middle. wondering create something like this.
    Text  3 :
     jquery accordion menu issue - user one answer open created jquery slide menu currently menu allows users one answer open time - would ideally like one open one time. anyone idea help would much appreciated. example: http: jsfiddle.net xmlgz
    Text  4 :
     view huge file webbrowser 150mb txt file server logs. probably php want view 100 lines time starting end file. user clicks pevious 100 lines something next set 100 loaded. would best way approach seen ajax chat widgets similar feature task
    
    Real tags:  ['java', 'android', 'jquery', 'php']
    Predicted tags:  ['java', 'android', 'jquery', 'php']
    


```python

```
