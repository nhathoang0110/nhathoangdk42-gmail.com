import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy import sparse, io


contents=[]

f = open("train.txt","r")

for line in f:
    line= line.rstrip()
    contents.append(line)

f.close()

print(len(contents))

vectorizer = CountVectorizer(min_df = 0.0005, max_df = 0.90)
X = vectorizer.fit_transform(contents)
io.mmwrite('train.mtx', X)
Y = vectorizer.get_feature_names()

# print(len(Y))
# print(X.shape)

# tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
# tfidf_transformer.fit(X)
# df_idf = pd.DataFrame(tfidf_transformer.idf_, index=Y,columns=["idf_weights"])
# print(df_idf)

# dense=X.todense()
# denselist=dense.tolist()
# df = pd.DataFrame(denselist, columns=Y)
# print(df)



f = open("vocab_train.txt", "w+")

vocab = ""
for i in Y:
    i = i + '\n'
    #print(i)
    vocab = vocab + i

f.write(vocab)
f.close()

