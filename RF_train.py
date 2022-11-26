import pylab
import numpy as np
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
import gym
import tensorflow as tf

# 그리드월드 예제에서의 REINFORCE 에이전트
class ReinforceAgent:
    def __init__(self, state_size, action_size):
        self.load_model = False
        # 상태와 행동의 크기 정의
        self.action_size = action_size
        self.state_size = state_size
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.model = self.build_model()
        self.optimizer = Adam(lr=self.learning_rate)
        self.states, self.actions, self.rewards = [], [], []

        if self.load_model:
            self.model.load_weights('./save_model/reinforce.h5')

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

    # 반환값 계산
    def discount_rewards(self):
        rewards = self.rewards
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # 한 에피소드 동안의 상태, 행동, 보상을 저장
    def append_sample(self, state, action, reward):
        self.states.append(state[0])
        self.rewards.append(reward)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)

    # 정책신경망 업데이트
    def train_model(self):
        discounted_rewards = np.float32(self.discount_rewards())
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        # 크로스 엔트로피 오류함수 계산
        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(model_params)
            policies = self.model(np.array(self.states))
            actions = np.array(self.actions)
            action_prob = tf.reduce_sum(actions * policies, axis=1)
            cross_entropy = - tf.math.log(action_prob + 1e-5)
            loss = tf.reduce_sum(cross_entropy * discounted_rewards)
            entropy = - policies * tf.math.log(policies)

        # 오류함수를 줄이는 방향으로 모델 업데이트
        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))
        self.states, self.actions, self.rewards = [], [], []
        return np.mean(entropy)


if __name__ == "__main__":
    EPISODES = 1000

    # 환경과 에이전트의 생성
    # 기존 코드에서 변경된 부분
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = ReinforceAgent(state_size, action_size)

    scores, episodes, entropies = [], [], []
    success = 0

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
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            agent.append_sample(state, action, reward)
            score += reward
            state = next_state

            if done:
                # 에피소드마다 정책신경망 업데이트
                entropy = agent.train_model()
                scores.append(score)
                episodes.append(e)
                entropies.append(entropy)
                print("episode: {:3d} | score: {:3f} | entropy: {:.3f}".format(e, score, entropy))

                # score 가 480 이상이면 연속적인 성공 횟수를 나타내는 success 에 1추가
                # 그렇지 않다면 연속해서 성공한 것이 아니므로 0으로 초기화
                if score > 480:
                    success += 1
                else:
                    success = 0

        # 10번 이상 좋은 score 를 냈다면 종료
        if success == 10:
            pylab.plot(episodes, scores, 'b')
            pylab.savefig("./save_graph/reinforce.png")
            agent.model.save_weights("./save_model/reinforce.h5")
            break

        # 10 번째 에피소드마다 policy table
        if e % 10 == 0:
            pass

        # 100 에피소드마다 학습 결과 출력 및 모델 저장
        if e % 100 == 0:
            pylab.plot(episodes, scores, 'b')
            pylab.savefig("./save_graph/reinforce.png")
            agent.model.save_weights("./save_model/reinforce.h5")


