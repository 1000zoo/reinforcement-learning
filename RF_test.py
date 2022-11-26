import copy
import pylab
import numpy as np
from environment import Env
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
import gym
import tensorflow as tf
import sys
from keras import backend as K
import os


# 그리드월드 예제에서의 REINFORCE 에이전트
class ReinforceAgent:
    def __init__(self, state_size, action_size):
        self.load_model = True
        # 상태와 행동의 크기 정의
        self.action_size = action_size
        self.state_size = state_size
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.model = self.build_model()
        self.optimizer = Adam(lr=self.learning_rate)
        self.states, self.actions, self.rewards = [], [], []

        if self.load_model:
            self.model.load_weights('./save_model/reinforce_good.h5')

    # 상태가 입력, 각 행동의 확률이 출력인 인공신경망 생성
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.summary()
        return model

    # 정책신경망으로 행동 선택
    def get_action(self, state):
        policy = self.model.predict(state, verbose=0)[0]
        return np.random.choice(self.action_size, 1, p=policy)[0]

if __name__ == "__main__":
    EPISODES = 10

    # 환경과 에이전트의 생성
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = ReinforceAgent(state_size, action_size)

    scores, episodes = [], []

    for e in range(EPISODES + 1):
        done = False
        score = 0
        # env 초기화
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            env.render()

            # 현재 상태에 대한 행동 선택
            action = agent.get_action(state)
            # 선택한 행동으로 환경에서 한 타임스탭 진행 후 샘플 수집
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            score += reward
            state = next_state

            if done:
                scores.append(score)
                episodes.append(e)
                print("episode: {:3d} | score: {:3f}".format(e, score))

            pylab.plot(episodes, scores, 'b')
            pylab.savefig("./save_graph/reinforce_test1.png")


