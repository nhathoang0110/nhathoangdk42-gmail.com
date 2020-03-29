import re 
import os
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer  
from nltk.tokenize import sent_tokenize, word_tokenize  

def docfile(file1):
    with open(file1,"rb") as f:
        contents = f.read()
        contents = contents.decode('utf-8','ignore')
    f.close()

    #arrays=contents.split()

    arrays=word_tokenize(contents)

    #loai bo ky tu dac biet va so
    for i in range(0,len(arrays)):
        arrays[i]= arrays[i].lower()
        arrays[i]= re.sub(r'[^a-z]', '', arrays[i])
    
    #loai bo stopword
    array1=[]
    for word in arrays:
        if(word) not in (stopwords.words('english')):
            array1.append(word)

    # steamming
    stemmer = PorterStemmer()

    array2=[]
    for word in array1:
        if(len(word)<10 and len(word)>2):
            array2.append(stemmer.stem(word))
    
    
    tf = np.unique(array2, return_counts = True)[1].tolist()         #bo lap
    value = np.unique(array2, return_counts = True)[0].tolist()

    str=' '.join(value)
    return str


path_train="/home/hoangntbn/Desktop/20192/project2/20news-bydate/20news-bydate-train"
path_test="/home/hoangntbn/Desktop/20192/project2/20news-bydate/20news-bydate-test"
FJoin = os.path.join


def solve(path):
    contents=""
    dirs = [FJoin(path, f) for f in os.listdir(path)]
    for i in range(0,len(dirs)):
        d=dirs[i]
        files = [FJoin(d,f) for f in os.listdir(d)]
        for j in range(0,len(files)):
            s= docfile(files[j])
            s= str(i)+"###" + s + "\n"
            contents=contents+s
    return contents    

contents_train=solve(path_train)
contents_test=solve(path_test)

file = open("train.txt", "w+")
file.write(contents_train)
file.close()

file = open("test.txt", "w+")
file.write(contents_test)
file.close()