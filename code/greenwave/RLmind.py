#定义了DQN类以及基础的网络类

import torch
import torch.nn as nn
import torch.Function as F
import numpy as np

class net(nn.Module):
    def __init__(self,):
        super(net, self).__init__()#继承来自net输入的内容
        self.f1 =nn.Linear(8,20)
        self.f1.weight = nn.init.normal(0,1.0)
        self.f2 =nn.Linear(20,2)
        self.f2.weight = nn.init.normal(0,1.0)#初始化参数

    def mish(self,x):
        return x*(torch.tanh(F.softplus(x)))#定义mish激活函数，性能比ReLU更优

    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.mish(x)
        return x

class DQN(memory_size,n_states):
    def __init__(self):
        self.estimator = net()
        self.target = net()
        self.memory = np.zeros((memory_size,(n_states+1)*2))#新建一个空的memory pool
        self.memory_counter = 0#记录有多少个tuple被存储
        self.optimizer = torch.optimizer.Adam()#Adam优化器
        self.LossF = nn.MSELoss()
    
    def choose_action(self,x):
        if np.random.random()<= epsilon:
            action = torch.max(self.estimator.forward(x),1)#根据网络选
        else:
            action = np.random.randint(0,n_actions,1)[1]#随机选
        return action

    def store_memory(self,s,a,s_,r):
        pass

    def update_network(self):
        pass
