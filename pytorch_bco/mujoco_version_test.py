import gym

env = gym.make('Reacher-v2')
print(env.observation_space.shape, env.action_space.shape)

for i in range(6):
    env.reset()
    rew = 0
    
    while True:
        _, r, done, _ = env.step(env.action_space.sample())
        
        rew += r
        
        if done==True:
            print('Ep %d: %.2f' % (i+1, rew))
            
            break


