import gym
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import load_model
import numpy as np

env = gym.make('CartPole-v0')
model = load_model('new_model2.h5')

for i_episode in range(20000):
    observation = env.reset()
    
    for t in range(200):
        env.render()
        best_action = np.argmax(model.predict(np.array([observation]))) #Runs the state and all actions through network and returns best.
        observation, reward, done, info = env.step(best_action)   
        if done:
          print("Episode finished after {} timesteps".format(t+1))
          break
