

```python
### *** French lemmatization and synsets with spacy and nltk *** ### 
```


```python
### Lemmatization ### 

# Note the wordnet included in the nltk corpus already includes French

import nltk 
from nltk.corpus import wordnet as wn 

# Example political text taken from 
# https://www.journaldemontreal.com/2019/05/14/front-commun-canada-mexique-contre-les-droits-de-douane-americains-1

text = """ OTTAWA | La ministre canadienne des Affaires étrangères, Chrystia Freeland, qui a reçu une délégation du gouvernement mexicain, a estimé mardi que les droits de douane américains sur l'aluminium et l'acier devaient être «abolis» pour «un vrai libre-échange» en Amérique du Nord."""

print(text)
```

     OTTAWA | La ministre canadienne des Affaires étrangères, Chrystia Freeland, qui a reçu une délégation du gouvernement mexicain, a estimé mardi que les droits de douane américains sur l'aluminium et l'acier devaient être «abolis» pour «un vrai libre-échange» en Amérique du Nord.
    


```python
### Tokenization ###

## nltk ## 
print("***nltk tokenization*** \n")

from nltk import word_tokenize
from nltk import sent_tokenize

word_tokens = [token for token in word_tokenize(text, language='french')] # tokenize by words
sent_tokens = [sent for sent in sent_tokenize(text, language='french')] # tokenize by sentence
print(word_tokens, "\n") 
print(sent_tokens, "\n")

"""Comments: 
    nltk seems to do the job just right."""

## spacy ## 

print("***spacy tokenization*** \n")

import spacy

nlp = spacy.load("fr_core_news_sm")
# NOTE: A more comprehensive model is available: fr_core_news_md

doc = nlp(text)

sp_word_tokens = [token.text for token in doc if token.text.isalpha()]
sp_sent_tokens = [sent for sent in doc.sents]
print(sp_word_tokens, "\n")
print(sp_sent_tokens, "\n")

""" Comments: 
    spacy also does the job """

```

    ***nltk tokenization*** 
    
    ['OTTAWA', '|', 'La', 'ministre', 'canadienne', 'des', 'Affaires', 'étrangères', ',', 'Chrystia', 'Freeland', ',', 'qui', 'a', 'reçu', 'une', 'délégation', 'du', 'gouvernement', 'mexicain', ',', 'a', 'estimé', 'mardi', 'que', 'les', 'droits', 'de', 'douane', 'américains', 'sur', "l'aluminium", 'et', "l'acier", 'devaient', 'être', '«', 'abolis', '»', 'pour', '«', 'un', 'vrai', 'libre-échange', '»', 'en', 'Amérique', 'du', 'Nord', '.'] 
    
    [" OTTAWA | La ministre canadienne des Affaires étrangères, Chrystia Freeland, qui a reçu une délégation du gouvernement mexicain, a estimé mardi que les droits de douane américains sur l'aluminium et l'acier devaient être «abolis» pour «un vrai libre-échange» en Amérique du Nord."] 
    
    ***spacy tokenization*** 
    
    ['OTTAWA', 'La', 'ministre', 'canadienne', 'des', 'Affaires', 'étrangères', 'Chrystia', 'Freeland', 'qui', 'a', 'reçu', 'une', 'délégation', 'du', 'gouvernement', 'mexicain', 'a', 'estimé', 'mardi', 'que', 'les', 'droits', 'de', 'douane', 'américains', 'sur', 'aluminium', 'et', 'acier', 'devaient', 'être', 'abolis', 'pour', 'un', 'vrai', 'en', 'Amérique', 'du', 'Nord'] 
    
    [ OTTAWA |, La ministre canadienne des Affaires étrangères, Chrystia Freeland, qui a reçu une délégation du gouvernement mexicain, a estimé mardi que les droits de douane américains sur l'aluminium et l'acier devaient être «abolis» pour «un vrai libre-échange» en Amérique du Nord.] 
    
    




    ' Comments: \n    spacy also does the job '




```python
### Lemmatization ### 

## nltk ##
print("**nltk lemmatization***\n")

from nltk.stem.snowball import FrenchStemmer

stemmer = FrenchStemmer() # instantiate the stemmer 

alpha_words = [w for w in word_tokens if w.isalpha()] # remove punctuation
nltk_fr_stems = [(word, stemmer.stem(word)) for word in alpha_words]

print(nltk_fr_stems, "\n") 

""" Comments: 
    - As it can be seen, nltk does a very poor job in lemmatizing French.
    - An alternative could be to implement our own stemmer with the 
        stem.Regexp() function, but it looks quite time consuming, 
        given all the irregularities of the language. In fact, implementing 
        a new module would be a better idea."""

## spacy lemmatization: ##
print("***spacy lemmatization***\n")

import spacy

nlp = spacy.load("fr_core_news_sm")

doc = nlp(text)
doc_lemmas = [(token.text, token.lemma_) for token in doc if token.text.isalpha()]
print(doc_lemmas)

print("Lemma: ", doc_lemmas[0][1])
""" Comments: 
    - Compared to nltk, spacy does an amazing job in lemmatizing French. """

```

    **nltk lemmatization***
    
    [('OTTAWA', 'ottaw'), ('La', 'la'), ('ministre', 'ministr'), ('canadienne', 'canadien'), ('des', 'de'), ('Affaires', 'affair'), ('étrangères', 'étranger'), ('Chrystia', 'chrysti'), ('Freeland', 'freeland'), ('qui', 'qui'), ('a', 'a'), ('reçu', 'reçu'), ('une', 'une'), ('délégation', 'déleg'), ('du', 'du'), ('gouvernement', 'gouvern'), ('mexicain', 'mexicain'), ('a', 'a'), ('estimé', 'estim'), ('mardi', 'mard'), ('que', 'que'), ('les', 'le'), ('droits', 'droit'), ('de', 'de'), ('douane', 'douan'), ('américains', 'américain'), ('sur', 'sur'), ('et', 'et'), ('devaient', 'dev'), ('être', 'être'), ('abolis', 'abol'), ('pour', 'pour'), ('un', 'un'), ('vrai', 'vrai'), ('en', 'en'), ('Amérique', 'amer'), ('du', 'du'), ('Nord', 'nord')] 
    
    ***spacy lemmatization***
    
    [('OTTAWA', 'ottawa'), ('La', 'le'), ('ministre', 'ministre'), ('canadienne', 'canadien'), ('des', 'un'), ('Affaires', 'affaire'), ('étrangères', 'étranger'), ('Chrystia', 'Chrystia'), ('Freeland', 'Freeland'), ('qui', 'qui'), ('a', 'avoir'), ('reçu', 'recevoir'), ('une', 'un'), ('délégation', 'délégation'), ('du', 'de'), ('gouvernement', 'gouvernement'), ('mexicain', 'mexicain'), ('a', 'avoir'), ('estimé', 'estimer'), ('mardi', 'mardi'), ('que', 'que'), ('les', 'le'), ('droits', 'droit'), ('de', 'de'), ('douane', 'douane'), ('américains', 'américain'), ('sur', 'sur'), ('aluminium', 'aluminium'), ('et', 'et'), ('acier', 'acier'), ('devaient', 'devoir'), ('être', 'être'), ('abolis', 'aboli'), ('pour', 'pour'), ('un', 'un'), ('vrai', 'vrai'), ('en', 'en'), ('Amérique', 'Amérique'), ('du', 'de'), ('Nord', 'nord')]
    Lemma:  ottawa
    




    ' Comments: \n    - Compared to nltk, spacy does an amazing job in lemmatizing French. '




```python
### Synsets ### 

## nltk ##

print("***nltk synsets***\n")

from nltk.corpus import wordnet as wn

# Note I decided to use alpha words and not stemmed given that the nltk stemmed words for 
# French are so imprecise. The result would be a bunch of empty synsets. 

# create a dictionary of synsets for each word
nltk_synsets_fr = {word: wn.synsets(word, lang='fra') for word in alpha_words}

def print_synsets(synset, num_synsets=10):
    """ expects a dictionary with synsets. 
        prints num_synsets number of synsets in the dictionary"""
    i = 0
    for key, val in synset.items():
        if i < num_synsets: 
            print("word : {}".format(key))
            print(val)
            i += 1
            print()
    
print_synsets(nltk_synsets_fr)

""" Comments: 
    Many words are missing synsets. """

## spacy + nltk ##

print("\n***spacy + nltk synsets***\n")

spnlkt_synsest_fr = {tup[0]: wn.synsets(tup[1], lang='fra') for tup in doc_lemmas}

print_synsets(spnlkt_synsest_fr) 

""" Comments: 
    - Better results but several entries are obviously wrong. 
    - We could filter things like articles and propositions since finding synsets for these 
        might not be very useful. """

```

    ***nltk synsets***
    
    word : OTTAWA
    [Synset('ottawa.n.03'), Synset('ottawa.n.01')]
    
    word : La
    [Synset('statistics.n.01'), Synset('statistics.n.01'), Synset('ladyship.n.01'), Synset('la.n.03'), Synset('la.n.03'), Synset('louisiana.n.01'), Synset('lanthanum.n.01'), Synset('lanthanum.n.01'), Synset('lanthanum.n.01')]
    
    word : ministre
    [Synset('minister.n.04'), Synset('indigo_bunting.n.01'), Synset('minister.v.02'), Synset('minister.v.01'), Synset('curate.n.01'), Synset('minister.n.03'), Synset('minister.n.02')]
    
    word : canadienne
    []
    
    word : des
    []
    
    word : Affaires
    []
    
    word : étrangères
    []
    
    word : Chrystia
    []
    
    word : Freeland
    []
    
    word : qui
    [Synset('world_health_organization.n.01')]
    
    
    ***spacy + nltk synsets***
    
    word : OTTAWA
    [Synset('ottawa.n.03'), Synset('ottawa.n.01')]
    
    word : La
    [Synset('lordship.n.02'), Synset('lordship.n.01'), Synset('lupus_erythematosus.n.01')]
    
    word : ministre
    [Synset('minister.n.04'), Synset('indigo_bunting.n.01'), Synset('minister.v.02'), Synset('minister.v.01'), Synset('curate.n.01'), Synset('minister.n.03'), Synset('minister.n.02')]
    
    word : canadienne
    [Synset('canadian.a.01'), Synset('canadian.n.02'), Synset('canadian.n.01'), Synset('french_canadian.n.01')]
    
    word : des
    [Synset('matchless.s.01'), Synset('one.s.05'), Synset('one.s.06'), Synset('one.s.04'), Synset('one.s.01'), Synset('one.s.01'), Synset('two.s.01'), Synset('three.s.01'), Synset('one.s.02'), Synset('one.n.02'), Synset('associate_in_nursing.n.01'), Synset('one.n.01'), Synset('one.n.01'), Synset('three.n.01'), Synset('sunday.n.01')]
    
    word : Affaires
    [Synset('matter.n.01'), Synset('business.n.04'), Synset('things.n.01'), Synset('case.n.12')]
    
    word : étrangères
    [Synset('foreign.a.02'), Synset('alien.s.02'), Synset('foreign.a.01'), Synset('alien.s.01'), Synset('estrange.v.02'), Synset('extraneous.s.03'), Synset('alien.v.01'), Synset('extraterrestrial_being.n.01'), Synset('foreigner.n.01'), Synset('foreigner.n.02'), Synset('stranger.n.01')]
    
    word : Chrystia
    []
    
    word : Freeland
    []
    
    word : qui
    [Synset('world_health_organization.n.01')]
    
    




    ' Comments: \n    - Better results but several entries are obviously wrong. \n    - We could filter things like articles and propositions since finding synsets for these \n        might not be very useful. '




```python
##  improved spacy + nltk ## 

print("***improved spacy+nltk***\n")

# create a filtered lemmas to exclude determiners (DET), adpositions(aka prepositions) (ADP), 
# punctuation (PUNCT), conjuctions (CONJ,CCONJ), numerals (NUM), symbols (SYM), spaces (NUM), 
# and non-alpha tokens. 
# Full list can be found at https://spacy.io/api/annotation

tags = ["DET","ADP","PUNCT","CONJ","CCONJ","NUM","SYM","SPACE"]
flt_doc_lemmas = [(token.text, token.lemma_) for token in doc if token.pos_ not in tags and token.text.isalpha()]

print("**filtered doc lemmas**\n")
print(flt_doc_lemmas)

# Now we will generate the synsets using these filtered lemmaS
print("\n**synsets**\n")

# list comprehension of lemmatized words
flt_synsets_fr = {tup[1]: wn.synsets(tup[1], lang='fra') for tup in flt_doc_lemmas}

print_synsets(flt_synsets_fr)

"""Comments: 
    - Improvement from the previous implementations. 
    - tags can be easily adjusted if one wishes to include them in the list comprehension. 
    - NOTE:  """
```

    ***improved spacy+nltk***
    
    **filtered doc lemmas**
    
    [('OTTAWA', 'ottawa'), ('ministre', 'ministre'), ('canadienne', 'canadien'), ('Affaires', 'affaire'), ('étrangères', 'étranger'), ('Chrystia', 'Chrystia'), ('Freeland', 'Freeland'), ('qui', 'qui'), ('a', 'avoir'), ('reçu', 'recevoir'), ('délégation', 'délégation'), ('gouvernement', 'gouvernement'), ('mexicain', 'mexicain'), ('a', 'avoir'), ('estimé', 'estimer'), ('mardi', 'mardi'), ('que', 'que'), ('droits', 'droit'), ('américains', 'américain'), ('aluminium', 'aluminium'), ('acier', 'acier'), ('devaient', 'devoir'), ('être', 'être'), ('abolis', 'aboli'), ('vrai', 'vrai'), ('Amérique', 'Amérique'), ('Nord', 'nord')]
    
    **synsets**
    
    word : ottawa
    [Synset('ottawa.n.03'), Synset('ottawa.n.01')]
    
    word : ministre
    [Synset('minister.n.04'), Synset('indigo_bunting.n.01'), Synset('minister.v.02'), Synset('minister.v.01'), Synset('curate.n.01'), Synset('minister.n.03'), Synset('minister.n.02')]
    
    word : canadien
    [Synset('canadian.a.01'), Synset('canadian.n.02'), Synset('canadian.n.01'), Synset('french_canadian.n.01')]
    
    word : affaire
    [Synset('matter.n.01'), Synset('business.n.04'), Synset('things.n.01'), Synset('case.n.12')]
    
    word : étranger
    [Synset('foreign.a.02'), Synset('alien.s.02'), Synset('foreign.a.01'), Synset('alien.s.01'), Synset('estrange.v.02'), Synset('extraneous.s.03'), Synset('alien.v.01'), Synset('extraterrestrial_being.n.01'), Synset('foreigner.n.01'), Synset('foreigner.n.02'), Synset('stranger.n.01')]
    
    word : Chrystia
    []
    
    word : Freeland
    []
    
    word : qui
    [Synset('world_health_organization.n.01')]
    
    word : avoir
    [Synset('beget.v.01'), Synset('give_birth.v.01'), Synset('have.v.12'), Synset('suffer.v.02'), Synset('contract.v.04'), Synset('grow.v.08'), Synset('get.v.03'), Synset('have.v.11'), Synset('receive.v.02'), Synset('catch.v.18'), Synset('get.v.20'), Synset('interview.v.01'), Synset('interview.v.03'), Synset('retention.n.01'), Synset('old.s.07'), Synset('deceive.v.02'), Synset('old.s.03'), Synset('drive.v.11'), Synset('get.v.14'), Synset('honest-to-god.s.01'), Synset('sleep_together.v.01'), Synset('take.v.35'), Synset('bring.v.04'), Synset('old.a.02'), Synset('draw.v.15'), Synset('old.a.01'), Synset('hold.v.03'), Synset('catch.v.24'), Synset('get.v.27'), Synset('scram.v.01'), Synset('experience.v.03'), Synset('have.v.01'), Synset('own.v.01'), Synset('have.v.09'), Synset('get.v.22'), Synset('receive.v.01'), Synset('accept.v.02'), Synset('obtain.v.01'), Synset('have.v.17'), Synset('get.v.21'), Synset('have.v.07'), Synset('deceive.v.01'), Synset('flim-flam.v.01'), Synset('have.v.02'), Synset('carry.v.21'), Synset('prevail.v.02'), Synset('embody.v.02'), Synset('carry.v.02'), Synset('have.v.10'), Synset('be.v.12'), Synset('carry.v.18'), Synset('minister.n.03'), Synset('rich_person.n.01'), Synset('property.n.01'), Synset('credit_side.n.01')]
    
    word : recevoir
    [Synset('beget.v.01'), Synset('suffer.v.02'), Synset('grow.v.08'), Synset('receive.v.08'), Synset('get.v.03'), Synset('receive.v.02'), Synset('catch.v.18'), Synset('receive.v.13'), Synset('get.v.20'), Synset('receive.v.06'), Synset('entertain.v.02'), Synset('welcome.v.02'), Synset('get.v.14'), Synset('receive.v.12'), Synset('get.v.19'), Synset('bring.v.04'), Synset('receive.v.05'), Synset('catch.v.07'), Synset('draw.v.15'), Synset('catch.v.24'), Synset('pick_up.v.09'), Synset('get.v.25'), Synset('catch.v.22'), Synset('catch.v.21'), Synset('get.v.22'), Synset('receive.v.01'), Synset('accept.v.09'), Synset('accept.v.02'), Synset('accept.v.05'), Synset('bear.v.06'), Synset('get.v.21'), Synset('receive.v.10'), Synset('meet.v.11')]
    
    




    'Comments: \n    - Improvement from the previous implementations. \n    - tags can be easily adjusted if one wishes to include them in the list comprehension. \n    - NOTE:  '




```python
### spaCy NER & visualization ### 

from spacy import displacy

for ent in doc.ents:
    print("[Entity:{}] [start:{}] [end:{}] [Label:{}]".format(ent.text, ent.start_char, ent.end_char, ent.label_))
    
displacy.render(doc, style="ent",jupyter=True)

# Another example taken from https://www.lemonde.fr/international/article/2019/05/11/au-mexique-la-double-crise-migratoire-tourne-au-casse-tete-pour-amlo_5460889_3210.html
text = """Chaque jour, les sans-papiers franchissent par centaines, parfois par milliers, le fleuve Suchiate, qui sépare le Guatemala du Mexique. Leur nombre est évalué à plus de 300 000 lors du premier trimestre 2019 par l’Institut national mexicain des migrations (INM), soit trois à quatre fois plus que les années précédentes. Du jamais-vu lié au nouveau phénomène des caravanes de Centraméricains, fuyant ensemble la misère et la violence de leurs pays d’origine. Ce tsunami migratoire provoque des goulots d’étranglement. A Tapachula, principale ville frontalière avec le Guatemala, les clandestins attendent en masse des permis de transit pour continuer leur route vers les Etats-Unis.""" 
doc = nlp(text)
displacy.render(doc, style="ent",jupyter=True)

```

    [Entity:OTTAWA] [start:1] [end:7] [Label:ORG]
    [Entity:Affaires étrangères] [start:37] [end:56] [Label:ORG]
    [Entity:Chrystia Freeland] [start:58] [end:75] [Label:PER]
    [Entity:Amérique du Nord] [start:262] [end:278] [Label:LOC]
    


<div class="entities" style="line-height: 2.5; direction: ltr"> 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone">
    OTTAWA
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 | La ministre canadienne des 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone">
    Affaires étrangères
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
, 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone">
    Chrystia Freeland
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">PER</span>
</mark>
, qui a reçu une délégation du gouvernement mexicain, a estimé mardi que les droits de douane américains sur l'aluminium et l'acier devaient être «abolis» pour «un vrai libre-échange» en 
<mark class="entity" style="background: #ff9561; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone">
    Amérique du Nord
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">LOC</span>
</mark>
.</div>



<div class="entities" style="line-height: 2.5; direction: ltr">Chaque jour, les sans-papiers franchissent par centaines, parfois par milliers, le fleuve 
<mark class="entity" style="background: #ff9561; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone">
    Suchiate
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">LOC</span>
</mark>
, qui sépare le 
<mark class="entity" style="background: #ff9561; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone">
    Guatemala
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">LOC</span>
</mark>
 du 
<mark class="entity" style="background: #ff9561; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone">
    Mexique
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">LOC</span>
</mark>
. Leur nombre est évalué à plus de 300 000 lors du premier trimestre 2019 par l’
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone">
    Institut national mexicain des migrations
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 (
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone">
    INM
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
), soit trois à quatre fois plus que les années précédentes. Du jamais-vu lié au nouveau phénomène des caravanes de 
<mark class="entity" style="background: #ff9561; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone">
    Centraméricains
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">LOC</span>
</mark>
, fuyant ensemble la misère et la violence de leurs pays d’origine. Ce tsunami migratoire provoque des goulots d’étranglement. A 
<mark class="entity" style="background: #ff9561; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone">
    Tapachula
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">LOC</span>
</mark>
, principale ville frontalière avec le 
<mark class="entity" style="background: #ff9561; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone">
    Guatemala
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">LOC</span>
</mark>
, les clandestins attendent en masse des permis de transit pour continuer leur route vers les 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone">
    Etats-Unis
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">MISC</span>
</mark>
.</div>

