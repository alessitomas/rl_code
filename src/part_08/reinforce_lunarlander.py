import gymnasium as gym
import torch
import pandas as pd

def train(env, y, lr, episodes):
    
    nn = torch.nn.Sequential(
        torch.nn.Linear(8,256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, env.action_space.n),
        torch.nn.Softmax(dim=-1)
    )
    # usa o Adam algorithm para otimização
    optim = torch.optim.Adam(nn.parameters(), lr=lr)

    statistics = []

    for i in range(episodes+1):
        (state, _) = env.reset()
        obs = torch.tensor(state, dtype=torch.float)
        done = False
        Actions, States, Rewards = [], [], []
        count_actions=0
        rewards=0

        while not done and count_actions < 1500:
            probs = nn(obs)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
            obs_, rew, done, truncated, _ = env.step(action)

            Actions.append(torch.tensor(action, dtype=torch.int))
            States.append(obs)
            Rewards.append(rew)

            obs = torch.tensor(obs_, dtype=torch.float)
            count_actions+=1
            rewards=rewards+rew
        
        if i % 100 == 0:
            print(f'Episode = {i}, Actions = {count_actions}, Rewards = {rewards}')
        
        statistics.append([i, count_actions, rewards])

        DiscountedReturns = []
        for t in range(len(Rewards)):
            G = 0.0
            for k,r in enumerate(Rewards[t:]):
                G += (y**k)*r
            DiscountedReturns.append(G)

        for State, Action, G in zip(States, Actions, DiscountedReturns):
            probs = nn(State)
            dist = torch.distributions.Categorical(probs=probs)
            log_prob = dist.log_prob(Action)

            # importante: aqui deve ser negativo pq eh um gradient ascendent
            loss = -log_prob*G

            optim.zero_grad()
            loss.backward()
            optim.step()
        
    return nn, statistics

#
# iniciando o treinamento
#
""" print('##### Treinando o modelo #####')

env = gym.make('LunarLander-v2')
lr = 0.000025
y = 0.9999
nn, statistics = train(env, y, lr, 30_000)
torch.save(nn, 'data/nn_ll.pt')
df = pd.DataFrame(statistics, columns = ['episode','actions','rewards'])
df.to_csv('results/statistics_lunarlander.csv') """

#
# Depois de treinado
#
print('##### Modelo treinado #####')

nn = torch.load('data/nn_ll.pt')
env = gym.make('LunarLander-v2', render_mode='human')
(state, _) = env.reset()
obs = torch.tensor(state, dtype=torch.float)
done = False

while not done: 
    probs = nn(obs)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample().item()
    obs_, rew, done, truncated, _info = env.step(action)    
    obs = torch.tensor(obs_, dtype=torch.float)



