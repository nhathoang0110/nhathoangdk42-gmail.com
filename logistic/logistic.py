
import numpy as np
import json

def sigmoid(x):
	return 1/(1+np.exp(-x))

class Logistic_Regresstion:
	def __init__(self):
		return

	def dot(self,x,w):
		result=w[-1]
		for key,value in x.items():
			result+=w[int(key)]*value
		return result

	def fit(self,X_train,Y_train,learning_rate=0.01,batch_size=32,max_epoch=100):
		self.labels=set(Y_train)
		self.W=np.array([np.random.randn(X_train.shape[1]) for label in self.labels])
		last_lose=1e9
		for ep in range(max_epoch):
			new_lose=0
			arr=np.array(range(X_train.shape[0]))
			np.random.shuffle(arr)
			X_train=X_train[arr]
			Y_train=Y_train[arr]
			total_batch=int(np.ceil(X_train.shape[0]/batch_size))
			for batch in range(total_batch):
				delta=[[0 for a in range(X_train.shape[1])] for label in self.labels]
				index=batch*batch_size
				X_sub=X_train[index:index+batch_size]
				Y_sub=Y_train[index:index+batch_size]
				dW=[]
				for label in self.labels:
					actual=[]
					for i in Y_sub:
						if label!=i:
							actual.append(0)
						else:
							actual.append(1)
					delta=sigmoid(X_sub.dot(self.W[label]))-actual
					new_lose+=delta.dot(delta)
					dW.append(learning_rate*delta.dot(X_sub))
				dW=np.array(dW)
				self.W=self.W-dW
			new_lose=new_lose/(X_train.shape[0]*len(self.labels))
			print('lose',new_lose)
			if np.abs(last_lose-new_lose)<=1e-4:
			 	print('stop at',ep)
			 	break
			last_lose=new_lose
		return self.W

	def predict(self,x):
		Max,argMax=0,0
		for label in self.labels:
			out=sigmoid(x.dot(self.W[label]))
			if Max<out:
				Max=out
				argMax=label
		return argMax

	def score(self,X_test,Y_test):
		count=0
		for i in range(len(Y_test)):
			vector,label=X_test[i],Y_test[i]
			predicted=self.predict(vector)
			if predicted==label :
				count+=1
		print('so luong du doan dung :',count)
		return count/len(Y_test)
			


with open('/home/hoangntbn/Desktop/20192/project2/20news-bydate/X_train.txt','r') as f:
  	X_train=np.array(json.load(f))
with open('/home/hoangntbn/Desktop/20192/project2/20news-bydate/Y_train.txt','r') as f:
  	Y_train=np.array(json.load(f))
with open('/home/hoangntbn/Desktop/20192/project2/20news-bydate/X_test.txt','r') as f:
  	X_test=np.array(json.load(f))
with open('/home/hoangntbn/Desktop/20192/project2/20news-bydate/Y_test.txt','r') as f:
  	Y_test=np.array(json.load(f))
LR=Logistic_Regresstion()
LR.fit(X_train,Y_train,learning_rate=0.01,batch_size=32,max_epoch=100)
print('Do chinh xac tren tap train :',LR.score(X_train,Y_train))
print('Do chinh xac tren tap test :',LR.score(X_test,Y_test))

