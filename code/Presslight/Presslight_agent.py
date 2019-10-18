#定义了DQN类以及基础的网络类

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class net(nn.Module):
    def __init__(self,n_states,n_actions):
        super(net, self).__init__()#继承来自net输入的内容
        self.f1 =nn.Linear(n_states,20)
        self.f1.weight.data.normal_(0,1)
        self.f2 =nn.Linear(20,n_actions)
        self.f2.weight.data.normal_(0,1)
        #self.f2.weight = nn.init.normal(0,1.0)#初始化参数

    def mish(self,x):
        return x*(torch.tanh(F.softplus(x)))#定义mish激活函数，性能比ReLU更优

    def forward(self, x):
        x = self.f1(x)
        x = self.mish(x)
        x = self.f2(x)
        return x

class DQN():
    def __init__(self,memory_size,n_states,n_actions,epsilon = 0.9,target_update_gap = 50,batch_size = 10,gamma = 0.8):
        self.estimator = net(n_states,n_actions)
        self.target = net(n_states,n_actions)
        self.memory = np.zeros((memory_size,(n_states+1)*2))#新建一个空的memory pool
        self.memory_counter = 0#记录有多少个tuple被存储
        self.optimizer = torch.optim.Adam(self.estimator.parameters())#Adam优化器
        self.LossF = nn.MSELoss()
        self.epsilon = epsilon
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.target_update_gap = target_update_gap
        self.n_actions = n_actions
        self.n_states = n_states
        self.gamma = gamma
        self.loss_records = []
        self.waitingtime = []
        self.ac_reward =[]
        self.reward = []

    def choose_action(self,x):
        if np.random.random()<= self.epsilon:
            action = torch.max(self.estimator.forward(x),1)[1].numpy()[0]#根据网络选
        else:
            action = np.random.randint(0,self.n_actions,1)[0]#随机选
        return action

    def store_memory(self,s,a,s_,r):
        #将 state_t,a_t,state_{t+1},reward_t存进一个tuple,位置的放置可以选择
        experience = np.hstack((s,a,r,s_))
        index = self.memory_counter%self.memory_size
        #index = np.random.randint(0,max(self.memory_size//2,self.memory_counter%self.memory_size))
        self.memory[index,:] = experience
        self.memory = self.memory[self.memory[:,self.n_states+1].argsort()]
        self.memory_counter += 1
        

    def update_network(self):
        #parameters transimitation
        if self.memory_counter>=self.memory_size and np.mod(self.memory_counter,self.target_update_gap) == 0:
            self.target.load_state_dict(self.estimator.state_dict())
        
        #draw the samples from the memory pool
        prob = list(np.ones([1,self.memory_size])[0]/self.memory_size)#uniformly draw samples
        #prob = [(i+1)*2/((self.memory_size+1)*self.memory_size) for i in range(self.memory_size)]#higher reward, higher prob
        sample_index = np.random.choice(self.memory_size,self.batch_size,p = prob,replace = False)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.n_states])
        b_a = torch.LongTensor(b_memory[:, self.n_states:self.n_states + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.n_states + 1:self.n_states + 2])
        b_s_ = torch.FloatTensor(b_memory[:, self.n_states+2:])

        #optimize the estimator network
        q_eval = self.estimator(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target(b_s_,).detach()  # detach from graph, don't backpropagate
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)  # shape (batch, 1)
        loss = self.LossF(q_eval, q_target)
        self.loss_records.append(loss.cpu().detach().numpy())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
