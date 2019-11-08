from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
import numpy as np
from joblib import load,dump
class Transition(object):
  s_curr = None
  a = None
  r_t = None
  s_next= None
  def __init__(self,s_curr,a,r,s_next):
    self.s_curr=s_curr
    self.a = a
    self.r = r
    self.s_next = s_next

model = Sequential([
    Dense(10,input_dim=2),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
    Dense(2),
    Activation('relu'),
    Dense(1)
])
#Random transitions
""" transitions = list()
for i in range(1,100):
  transitions.append(Transition(np.random.rand(1,4),np.random.randint(1,2),0.1,np.random.rand(1,4)))
X = np.zeros([100,4])
Y = np.zeros([100])
for i,trans in enumerate(transitions):
  X[i,:] = trans.s_curr
  Y[i] = trans.r + 0.1*model.predict(trans.s_curr)

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])
model.fit(X,Y,epochs=5,batch_size=10) """
dump(model,'model.joblib')
print(model.predict((np.array([[1,10],[2,3]]))))


