import pylab
import random
import numpy as np
from collections import deque

from environment import Env

from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

class DQN(Sequential):
    def __init__(self, state_size, action_size, learning_rate):
        super().__init__()
        self.add(Dense(30, input_dim=state_size, activation='relu'))
        self.add(Dense(30, activation='relu'))
        self.add(Dense(action_size, activation='linear'))
        self.summary()
        self.compile(loss='mse', optimizer=Adam(lr=learning_rate))

class DeepSARSAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.discount_factor = 0.99
        self.lr = 0.001
        self.epsilon = 1
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000

        self.memory = deque(maxlen=2000)

        self.model = DQN(self.state_size, self.action_size, self.lr)
        self.target_model = DQN(self.state_size, self.action_size, self.lr)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            # state = np.float32(state)
            q_values = self.model.predict(state, verbose=0)
            return np.argmax(q_values[0])


    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 메모리에서 배치 크기만큼 무작위로 샘플 추출
        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        # 현재 상태에 대한 모델의 큐함수
        # 다음 상태에 대한 타깃 모델의 큐함수
        target = self.model.predict(states, verbose=0)
        target_val = self.target_model.predict(next_states, verbose=0)


        # 벨만 최적 방정식을 이용한 업데이트 타깃
        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * (
                    np.amax(target_val[i]))

        self.model.fit(states, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)


if __name__ == "__main__":
    env = Env(render_speed=0.00001)
    state_size = 15
    action_space = [0, 1, 2, 3, 4]
    action_size = len(action_space)
    agent = DeepSARSAgent(state_size, action_size)

    scores, episodes = [], []

    for episode in range(201):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        time = 0

        while not done:
            action = agent.get_action(state)

            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            next_action = agent.get_action(next_state)

            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            score += reward
            state = next_state
            time += 1
            if time > 200:
                done = True

            agent.append_sample(state, action, score, next_state, done)

            if done:
                print("episode: {:3d} | score: {:3f} | epsilon: {:.3f} | memory : {:3d}".format(
                      episode, score, agent.epsilon, len(agent.memory)))
                scores.append(score)
                episodes.append(episode)
                agent.update_target_model()

        if episode % 100 == 0:
            agent.model.save_weights('save_model/model', save_format='tf')
            pylab.plot(episodes, scores, 'b')
            pylab.xlabel("episode")
            pylab.ylabel("score")
            pylab.savefig("./graph.png")
    # agent.model.save_weights('model', save_format='tf')

