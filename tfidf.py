import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy import sparse, io


contents=[]

f = open("train.txt","r")
dem=0
for line in f:
    line= line.rstrip()
    contents.append(line)
    dem=dem+1
    if dem==5:
        break

f.close()

print(len(contents))

vectorizer = CountVectorizer(min_df = 0.0005, max_df = 0.90)
X = vectorizer.fit_transform(contents)
# io.mmwrite('train.mtx', X)
Y = vectorizer.get_feature_names()


tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(X)
# df_idf = pd.DataFrame(tfidf_transformer.idf_, index=Y,columns=["idf_weights"])
# print(df_idf)

# dense=X[0].todense()
# denselist=dense.tolist()
# df = pd.DataFrame(denselist, columns=Y)
# print(df)

count_vector=vectorizer.transform(contents)
tf_idf_vector=tfidf_transformer.transform(count_vector)
first=tf_idf_vector[0]

df = pd.DataFrame(first.T.todense(), index=Y, columns=["tfidf"])
df.sort_values(by=["tfidf"],ascending=False)
print(df)






