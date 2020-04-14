import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer
from scipy import sparse, io


contents=[]

f = open("train1.txt","r")
labels=""
for line in f:
    line= line.rstrip()
    line=line.split("###")
    contents.append(line[1])
    labels = labels + line[0] + "\n"
    # contents.append(line)

f.close()
print(type(contents))
print(len(contents))
f = open("label.txt", "w+")
f.write(labels)
f.close()

tfidf_vectorizer = TfidfVectorizer(min_df = 0.0005, max_df = 0.90)
train = tfidf_vectorizer.fit_transform(contents)
print(train.shape)
# io.mmwrite('train.mtx', train)
voc = tfidf_vectorizer.get_feature_names()
print(len(voc))


first_vector_tfidfvectorizer=train[0]
df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=voc, columns=["tfidf"])
df=df.sort_values(by=["tfidf"],ascending=False)
print(df)
# #tinh idf
# tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
# tfidf_transformer.fit(train)
# df_idf = pd.DataFrame(tfidf_transformer.idf_, index=Y,columns=["idf_weights"])
# df_idf=df_idf.sort_values(by=['idf_weights'])
# print(df_idf)
# #tfidf
# count_vector=cv.transform(contents)
# tfidf_vector=tfidf_transformer.transform(count_vector)
# first_document_vector=tfidf_vector[0]
# df = pd.DataFrame(first_document_vector.T.todense(), index=Y, columns=["tfidf"])
# df=df.sort_values(by=["tfidf"],ascending=False)
# # print(df)

# f = open("vocab_train.txt", "w+")

# vocab = ""
# for i in voc:
#     i = i + '\n'
#     vocab = vocab + i

# f.write(vocab)
# f.close()


# #####################################

f = open("test.txt", "r")
contents_test=[]

labels_test = ""
for line in f:
    line = line.rstrip()
    line = line.split("###")
    contents_test.append(line[1])
    labels_test = labels_test + line[0] + "\n"

f.close()

f = open("label_test.txt", "w+")
f.write(labels_test)
f.close()



tfidf_vectorizer = TfidfVectorizer(min_df = 0.0005, max_df = 0.990)
test = tfidf_vectorizer.fit_transform(contents_test)
print("test.shape: ", test.shape)
io.mmwrite('test.mtx', test)
voc_test =  tfidf_vectorizer.get_feature_names()
print(len(voc_test))


f = open("vocab_test.txt", "w+")
vocab = ""
for i in voc_test:
    i = i + '\n'
    #print(i)
    vocab = vocab + i

f.write(vocab)
f.close()

