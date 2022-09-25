import gym
import time

env = gym.make("CartPole-v1")
observation = env.reset()

for _ in range(1000):
    env.render()
    action = env.action_space.sample()

    observation, reward, done, info = env.step(action)

    if done:
        observation = env.reset()
    
    print("="*30)
    print("count:", _)
    print("observation:", observation, "/ type:", type(observation))
    print("reward:", reward, "/ type:", type(reward))
    print("done:", done, "/ type:", type(done))
    print("info:", info, "/ type:", type(info))
    print("="*30)
    time.sleep(0.1)

env.close()