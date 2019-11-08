import gym
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
env = gym.make('CartPole-v0')
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
    Dense(10,input_dim=5),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
    Dense(1),
])
model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])
epsilon = 0.2
gamma = 0.1


replay_memory = set()
#for i in range(1,1000):
  #replay_memory.add(Transition(np.random.rand(1,4),np.random.randint(0,1),0.1,np.random.rand(1,4)))

for i_episode in range(1000):
    observation = env.reset()
    
    for t in range(1000):
        #env.render()
        best_action = np.random.randint(0,1) if (np.random.random() < epsilon) else np.argmax(model.predict(np.array([np.append(observation,1),np.append(observation,2)]))) #Runs the state and all actions through network and returns best.
        new_observation, reward, done, info = env.step(best_action)
        #print(new_observation)
        replay_memory.add(Transition(observation,best_action,reward,new_observation))
        observation = new_observation
        
       
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            #print(len(replay_memory))
            if(len(replay_memory) > 200):
              print('Updating model..')
              X_replay = np.zeros([200,5])
              Y_replay = np.zeros([200])
              for i in range(1,200):
                  replay = replay_memory.pop()
                  #print(replay.s_curr)
                  X_replay[i,:] = np.append(replay.s_curr,replay.a)
                  followup_best_action = np.argmax(model.predict(np.array([np.append(replay.s_next,1),np.append(replay.s_next,2)]))) #Which action gives best value given the next state

                  if(np.abs(replay.s_next[0])>4.8): #If it was a terminal state
                    Y_replay[i] = replay.r
                  else:
                    Y_replay[i] = replay.r+gamma*model.predict(np.array([np.append(replay.s_next,followup_best_action)]))

              model.fit(X_replay,Y_replay,epochs=5,batch_size=20) #Model is updated
            break

    if(i_episode == 10): #Flush replay memory occasionally
      replay_memory = set()
env.close()