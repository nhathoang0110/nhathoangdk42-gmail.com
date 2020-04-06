import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy import sparse, io


contents=[]

f = open("train1.txt","r")
labels=""
for line in f:
    line= line.rstrip()
    line=line.split("###")
    # print(line[0])
    contents.append(line[1])
    labels = labels + line[0] + "\n"
    # contents.append(line)

f.close()
print(type(contents))
print(len(contents))
f = open("label.txt", "w+")
f.write(labels)
f.close()

cv = CountVectorizer(min_df = 0.0005, max_df = 0.90)
X = cv.fit_transform(contents)
io.mmwrite('train.mtx', X)
Y = cv.get_feature_names()

#tinh idf
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(X)
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=Y,columns=["idf_weights"])
df_idf=df_idf.sort_values(by=['idf_weights'])
print(df_idf)


#tfidf
count_vector=cv.transform(contents)
tfidf_vector=tfidf_transformer.transform(count_vector)

first_document_vector=tfidf_vector[0]
df = pd.DataFrame(first_document_vector.T.todense(), index=Y, columns=["tfidf"])
df=df.sort_values(by=["tfidf"],ascending=False)
# print(df)

f = open("vocab_train.txt", "w+")

vocab = ""
for i in Y:
    i = i + '\n'
    vocab = vocab + i

f.write(vocab)
f.close()

