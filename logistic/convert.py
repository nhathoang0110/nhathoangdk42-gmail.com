import numpy as np
import json

def load_data_to_sparse(data_path,vocab_size,fname):
	with open(data_path,'r') as f:
		lines=f.read().splitlines()
	X,Y=[],[]
	for line in lines:
		lp=line.split('<fff>')
		label,vector=int(lp[0]),lp[-1]
		dense_vector=[(int(i.split(':')[0]),float(i.split(':')[1])) for i in vector.split()]
		sparse_vecter=[0 for i in range(vocab_size)]
		sparse_vecter.append(1)
		for key,value in dense_vector:
			sparse_vecter[key]=value
		X.append(sparse_vecter)
		Y.append(label)
	with open('/home/hoangntbn/Desktop/20192/project2/20news-bydate/X_'+fname+'.txt','w') as f:
		json.dump(X,f)
	with open('/home/hoangntbn/Desktop/20192/project2/20news-bydate/Y_'+fname+'.txt','w') as f:
		json.dump(Y,f)
	return np.array(X),np.array(Y)


with open('/home/hoangntbn/Desktop/20192/project2/20news-bydate/words_idf.txt','r') as f:
  	vocab_size=len(f.read().splitlines())
X_train,Y_train=load_data_to_sparse('/home/hoangntbn/Desktop/20192/project2/20news-bydate/tf_idf_vector.txt',vocab_size,'train')
X_test,Y_test=load_data_to_sparse('/home/hoangntbn/Desktop/20192/project2/20news-bydate/test_tf_idf_vector.txt',vocab_size,'test')