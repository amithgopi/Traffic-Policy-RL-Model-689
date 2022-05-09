import os
import numpy as np
import sys

import torch.nn as nn
import torch

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
cpu_device = torch.device('cpu')

class ValueModel(nn.Module):
    def __init__(self, input_dim, width, output_dim, num_layers):
        super().__init__()
        self.activation = torch.tanh
        inputs = nn.Linear(input_dim, width)
        self.layers = nn.ModuleList()
        self.layers.append(inputs)
        for _ in range(num_layers-2):
            self.layers.append(nn.Linear(width, width))
        self.outputLayer = nn.Linear(width, 1)
        self.outputLayer.weight.data.mul_(0.1)
        self.outputLayer.bias.data.mul_(0.0)
    
    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.outputLayer(x)

class PolicyModel(nn.Module):
    def __init__(self, input_dim, width, output_dim, num_layers):
        super().__init__()
        self.activation = torch.tanh
        inputs = nn.Linear(input_dim, width)
        self.layers = nn.ModuleList()
        self.layers.append(inputs)
        for _ in range(num_layers-2):
            self.layers.append(nn.Linear(width, width))
        self.outputLayer = nn.Linear(width, output_dim)
        self.outputLayer.weight.data.mul_(0.1)
        self.outputLayer.bias.data.mul_(0.0)
    
    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        # print(x.size())
        out = torch.softmax(self.outputLayer(x), dim=0)
        # print(out.size())
        return out

    def get_log_prob(self, x, actions):
        action_prob = self.forward(x)
        # print(action_prob, actions)
        probs =  torch.log(action_prob.gather(1, actions.long()))
        # print(probs.size())
        return probs
    
    def select_action(self, x):
        # print(x.size())
        action_prob = self.forward(x)
        action = action_prob.multinomial(1)
        return action

class TrainModel:
    def __init__(self, num_layers, width, batch_size, learning_rate, gamma, input_dim, output_dim):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._gamma = gamma
        self._model = self._build_model(num_layers, width)
        self.policy_model = PolicyModel(self.input_dim, width, output_dim, num_layers)
        # self.criterion = nn.MSELoss()
        self.optimizer_policy = torch.optim.Adam(self._model.parameters(), lr=learning_rate)
        self.optimizer_value = torch.optim.Adam(self.policy_model.parameters(), lr=learning_rate)



    def _build_model(self, num_layers, width):
        """
        Build and compile a fully connected deep neural network
        """
        model = ValueModel(self.input_dim, width, self.output_dim, num_layers)
        print(model)
        return model
    
    def select_action(self, state):
        state: torch.Tensor = torch.tensor(state).float().to(device)
        return self.policy_model.select_action(state).detach().to(cpu_device).numpy()
    
    def compute_returns(self, rewards, masks):
        discountedReward = 0
        returns = []
        for step in reversed(range(len(rewards))):
            discountedReward = rewards[step] + self._gamma * discountedReward * masks[step]
            returns.append(discountedReward)
        returns.reverse()
        return torch.tensor(returns)
    
    def estimate_actor_loss(self, returns, log_policies, values, masks):
        # traj_indexes = [-1] + [idx for idx in range(len(masks)) if masks[idx] == 0]
        trajectoryCount = len(masks) - np.count_nonzero(masks)

        # returns = returns.unsqueeze(1)
        advantages = returns - values.detach()
        # print(returns)
        # print(values)
        critic_loss = -(advantages*log_policies)
        # print("traj", trajectoryCount, "crit loss", critic_loss.size())
        critic_loss = critic_loss.sum() #/ trajectoryCount
        
        return critic_loss

    def train(self, states, actions, nextStates, rewards):
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).float().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        # rewards = rewards + rewards.min().abs()
        masks = np.ones(len(states))
        
        # masks[-1] = 0
        masks = torch.from_numpy(masks).float().to(device)
        returns = self.compute_returns(rewards, masks).unsqueeze(1)
        returns = (returns - returns.mean()) / returns.std()
        log_policies = self.policy_model.get_log_prob(states, actions)
        values = self._model(states)

        actor_loss = self.estimate_actor_loss(returns, log_policies, values, masks)
        # print("ret-val", returns.size(), values.size(), (returns - values).pow(2).size())
        critic_loss = (returns - values).pow(2).mean()
        print(critic_loss, actor_loss)
        self.optimizer_policy.zero_grad()
        self.optimizer_value.zero_grad()
        critic_loss.backward()
        actor_loss.backward()
        self.optimizer_policy.step()
        self.optimizer_value.step()


        


    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        state: torch.Tensor = torch.tensor(state).float().to(device)
        # print(state, state.size())
        return self._model(state).detach().to(cpu_device).numpy()


    def predict_batch(self, states):
        """
        Predict the action values from a batch of states
        """
        states = torch.tensor(states).float().to(device)
        # print("batch", states, states.size())
        return self._model(states).detach().to(cpu_device).numpy()


    def train_batch(self, states, q_sa, actions):
        """
        Train the nn using the updated q-values
        """
        actions = torch.tensor(actions).unsqueeze(1).to(torch.int64).to(device)
        # print(actions.type(), actions.size())
        states = torch.tensor(states).float().to(device)
        currentValues = self._model(states).gather(1, actions)
        targetValues = torch.tensor(q_sa).unsqueeze(1).float().to(device)
        self.optimizer.zero_grad()
        # print(currentValues.size(), targetValues.size())
        loss = self.criterion(currentValues, targetValues)
        # print(currentValues, targetValues)
        # print(loss)
        loss.backward()
        self.optimizer.step()


    def save_model(self, path):
        """
        Save the current model in the folder as h5 file and a model architecture summary as png
        """
        torch.save(self.policy_model, os.path.join(path, 'trained_model.h5'))


    @property
    def input_dim(self):
        return self._input_dim


    @property
    def output_dim(self):
        return self._output_dim


    @property
    def batch_size(self):
        return self._batch_size


class TestModel:
    def __init__(self, input_dim, model_path):
        self._input_dim = input_dim
        self._model = self._load_my_model(model_path)
        self._model.eval()


    def _load_my_model(self, model_folder_path):
        """
        Load the model stored in the folder specified by the model number, if it exists
        """
        print("Loading Model from File")
        model_file_path = os.path.join(model_folder_path, 'trained_model.h5')
        
        if os.path.isfile(model_file_path):
            loaded_model = torch.load(model_file_path)
            return loaded_model
        else:
            sys.exit("Model number not found")

    def select_action(self, state):
        state: torch.Tensor = torch.tensor(state).float().to(device)
        return self._model.select_action(state).detach().to(cpu_device).numpy()
    # def predict_one(self, state):
    #     """
    #     Predict the action values from a single state
    #     """
    #     with torch.no_grad():
    #         state: torch.Tensor = torch.tensor(state).float().to(device)
    #         prediction = self._model(state).detach().to(cpu_device).numpy()
    #     return prediction



    @property
    def input_dim(self):
        return self._input_dim