import gym
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import layers
import numpy as np
import random
from collections import deque
env = gym.make('CartPole-v0')
class Transition(object):
  s_curr = None
  a = None
  r_t = None
  s_next= None
  done= False
  def __init__(self,s_curr,a,r,s_next,done):
    self.s_curr=s_curr
    self.a = a
    self.r = r
    self.s_next = s_next
    self.done = done

nb_actions = env.action_space.n
state_shape = env.observation_space.shape
nbr_states = 4

model = Sequential()
model.add(layers.Dense(24, activation='relu', input_shape=state_shape))
model.add(layers.Dense(24, activation='relu'))
model.add(layers.Dense(nb_actions,activation='linear'))

model.compile(loss='mse', optimizer=Adam(lr=0.001))

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.95
gamma = 0.99


replay_memory = deque(maxlen=2000)

for i_episode in range(2000):
    observation = env.reset()
    max_pos = -10
    for t in range(1000):
        #env.render()
        if i_episode%50==0:
            env.render()
        best_action = env.action_space.sample() if (np.random.random() < epsilon) else np.argmax(model.predict(np.array([observation])))
        new_observation, reward, done, info = env.step(best_action)
        if(new_observation[0] > max_pos):
          max_pos = new_observation[0]
        #if new_observation[0] >= 0.5:
                #reward += 10
        replay_memory.append(Transition(observation,best_action,reward,new_observation,done))
        observation = new_observation       
        if done:
            
          #print(len(replay_memory))
          #if(max_pos>0.5):
          print("episode: {}/{}, score: {}"
                      .format(i_episode, 2000, t))
          if(i_episode%400 == 0):
            model.save('mountain_car.h5')
          break

    if(len(replay_memory)>32):
      minibatch = random.sample(replay_memory,32)

      for replay in minibatch:
          state = np.array([replay.s_curr])
          q_vals = model.predict(np.array([replay.s_next]))
          followup_best_action = np.argmax(q_vals) #Which action gives best value given the next state

          if(replay.done):
            target = replay.r
          else:
            target = replay.r+gamma*q_vals[0][followup_best_action]
          target_full = model.predict(np.array([replay.s_curr])) #What the model prediction would be
          target_full[0][replay.a] = target #Update the value for the action that would give the best q-value for the next state
       
          model.fit(state,target_full,epochs=1,verbose=False) #Model is updated
      epsilon *=epsilon_decay
        

    #if(i_episode == 2000): #Flush replay memory occasionally
      #replay_memory = set()
model.save('mountain_car.h5')
env.close()
