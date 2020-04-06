from nltk.stem import PorterStemmer  
from nltk.tokenize import sent_tokenize, word_tokenize  
import numpy as np
stemmer = PorterStemmer()

example_words = ["wait", "a", "waited","waiting","waits","played","playing",]

a= []

for w in example_words:
    a.append(stemmer.stem(w))

print(1)
print(a)

sentence="Hello Guru99, You have to build a very good sites and I love visiting your site"

words=word_tokenize(sentence)
a1=[]
for w in words:
    a1.append(stemmer.stem(w))
print(a1)

tf = np.unique(a1, return_counts = True)[1].tolist()         #bo lap
value = np.unique(a1, return_counts = True)[0].tolist()

print(tf)
print(value)