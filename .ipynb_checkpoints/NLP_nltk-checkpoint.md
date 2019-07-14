<p align="center"> 
<img src="https://github.com/emunozlorenzo/MasterDataScience/blob/master/img/image2.png">
</p>

# NLP: NLTK
___

![alt text](https://github.com/emunozlorenzo/MasterDataScience/blob/master/img/icon2.png "Logo Title Text 1") [Eduardo Muñoz](https://www.linkedin.com/in/eduardo-mu%C3%B1oz-lorenzo-14144a144/)

## Installation

```python
pip3 install nltk
```

## 1. Accessing Text Corpora and Lexical Resources

### 1.1 Unannotated Corpora

```python
import nltk
nltk.download('book')
nltk.download('cess_esp')
```
```python
from nltk.corpus import gutenberg
# Lista de Libros del Proyecto Gutenberg
print(gutenberg.fileids())

# gutenberg.raw recupera el texto como una única cadena de caracteres.
print(gutenberg.raw(fileids='shakespeare-caesar.txt'))

# gutenberg.words recupera el texto tokenizado en palabras. El método devuelve una lista palabras.
print(gutenberg.words(fileids='shakespeare-caesar.txt'))

# gutenberg.sents recupera el texto segmentado por oraciones. El método devuelve una lista de oraciones. Cada oración es a su vez una lista de palabras.
print(gutenberg.sent(fileids='shakespeare-caesar.txt'))

# gutenberg.paras recupera el texto segmentado por párrafos. El método devuelve una lista de párrafos. Cada párrafo es una lista de oraciones, cada oración es a su vez una lista de palabras.
print(gutenberg.paras(fileids='shakespeare-caesar.txt'))
```
```python
# la versión en crudo de este corpus contiene información morfosintáctica.
from nltk.corpus import cess_esp
```

### 1.2 Annotated Corpora

El Corpus de Brown fue el primer gran corpus orientado a tareas de NLP. Desarrollado en la Universidad de Brown, contiene más de un millón de palabras provenientes de 500 fuentes.  La principal catacterística de este corpus es que sus textos están categorizados por género. 

```python
from nltk.corpus import brown
# Brown está formado por 500 documentos
print(len(brown.fileids()))
# Categories: adventure, belles_lettres, editorial, fiction, government...
print(brown.categories())
# News: words
news_words = brown.words(categories='news')
# Scifi: sentences
scifi_sents = brown.sents(categories='science_fiction')
```

### 1.3 Frequency Distribution

```python
from nltk import ConditionalFreqDist
modals = 'can could would should must may might'.split()
modals_cfd = ConditionalFreqDist((category, word) 
                                for category in brown.categories() 
                                for word in brown.words(categories=category))
# Table
modals_cfd.tabulate(conditions=brown.categories(), samples=modals)

# imprimo solo algunos verbos modales para la categoría fiction
modals_cfd.tabulate(conditions=['fiction'], samples=['can', 'should', 'would'])
```

### 1.4 Tagged Words, Sentences and Paragraphs

El corpus de Brown no solo está categorizado, también está anotado con información morfológica. Para acceder a la versión anotada del corpus, podemos utilizar los métodos: 

*brown.tagged_words*, 
*brown.tagged_sents*, 
*brown.tagged_ paras*

```python
scifi_tagged_words = brown.tagged_words(categories='science_fiction')
print(scifi_tagged_words[:50])
```
```python
# Frequency Distribution: sustantivos en singular NN, verbos en presente VB, verbos en pasado simple VBD, participios pasados VBN, adjetivos JJ, preposiciones IN, y artículos AT. 
tags_cfd = ConditionalFreqDist(
                                (category, item[1])
                                for category in brown.categories()
                                for item in brown.tagged_words(categories=category)
                              )
tags_cfd.tabulate(conditions=brown.categories(), 
                  samples='NN VB VBD VBN JJ IN AT'.split())
```
