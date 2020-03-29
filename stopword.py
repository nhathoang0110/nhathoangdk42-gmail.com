import nltk
import numpy as np
from nltk.corpus import stopwords

array = ['aaa', 'a', 'hello','an','the','hi','hi']

print(1)
array1 = []

for word in array:
    if(word) not in (stopwords.words('english')):
        array1.append(word)


tf = np.unique(array1, return_counts = True)[1].tolist()
value = np.unique(array1, return_counts = True)[0].tolist()

print(tf)
print(value)
str = ' '.join(value)

print(str)