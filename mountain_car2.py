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

nb_actions = env.action_space.n
model = Sequential([
    Dense(16,input_dim=4),
    Activation('relu'),
    Dense(nb_actions),
    Activation('linear')
])
model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])

target = Sequential([
    Dense(16,input_dim=4),
    Activation('relu'),
    Dense(nb_actions),
    Activation('linear')
])
target.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])
epsilon = 0.3
gamma = 0.05
replay_memory = set()


for i_episode in range(10000):
    observation = env.reset()
    
    for t in range(200):
        #env.render()
        print(np.array([observation]))
        best_action = env.action_space.sample() if (np.random.random() < epsilon) else np.argmax(model.predict(np.array([observation]))) #Runs the state and all actions through network and returns best.
        new_observation, reward, done, info = env.step(best_action)

        replay_memory.add(Transition(observation,best_action,reward,new_observation))
        observation = new_observation

        replay = replay_memory.pop()
        X_replay = np.array([replay.s_curr])
       
        if(done): #If it was a terminal state
          Y_replay= np.array([[replay.r]])
        else:
          Y_replay = np.array(replay.r + gamma*target.predict(np.array([replay.s_next])))
        print(X_replay)
        print(Y_replay)
        target.fit(X_replay,Y_replay) #Model is updated
        replay_memory.add(replay)    
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            #print(len(replay_memory))
            break
    if(i_episode%200 == 0):
      model.set_weights(target.get_weights()) #Update actual model with target weights
    
          

      
      

    #if(i_episode == 2000): #Flush replay memory occasionally
      #replay_memory = set()
env.close()