# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 00:31:26 2020

@author: satya
"""

import nltk 
#nltk.download()
#nltk.download('punkt')
#nltk.download('wordnet')

paragraph="""The WordNetLemmatizer may be the culprit. Wordnet needs to read from several files to work. There are lots of file access OS-level stuff that may hinder performance. Consider using another lemmatizer, see if the hard drive of the slow computer is faulty or try defragmenting it (if on windows)"""

import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
LM=WordNetLemmatizer()
PS=PorterStemmer()
processed=re.sub('[^a-zA-Z]','',paragraph)



sentence=nltk.sent_tokenize(paragraph) 
sentence=[nltk.word_tokenize(i) for i in sentence]
for i in range(len(sentence)):
    sentence[i]=[PS.stem(word) for word in sentence[i] if word not in stopwords.words('english')]

from sklearn.feature_extraction.text import TfidfVectorizer
TF=TfidfVectorizer()
transformed=TF.fit_transform(sentence[3])
transformed.toarray()