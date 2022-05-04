import os
import numpy as np
import sys

import torch.nn as nn
import torch

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
cpu_device = torch.device('cpu')

class PolicyModel(nn.Module):
    def __init__(self, input_dim, width, output_dim, num_layers):
        super().__init__()
        self.activation = torch.relu
        inputs = nn.Linear(input_dim, width)
        self.layers = nn.ModuleList()
        self.layers.append(inputs)
        for _ in range(num_layers-2):
            self.layers.append(nn.Linear(width, width))
        self.outputLayer = nn.Linear(width, output_dim)
    
    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.outputLayer(x)

class TrainModel:
    def __init__(self, num_layers, width, batch_size, learning_rate, input_dim, output_dim):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._model = self._build_model(num_layers, width)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)


    def _build_model(self, num_layers, width):
        """
        Build and compile a fully connected deep neural network
        """
        model = PolicyModel(self.input_dim, width, self.output_dim, num_layers)
        print(model)
        return model
    

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
        torch.save(self._model, os.path.join(path, 'trained_model.h5'))


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


    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        with torch.no_grad():
            state: torch.Tensor = torch.tensor(state).float().to(device)
            prediction = self._model(state).detach().to(cpu_device).numpy()
        return prediction



    @property
    def input_dim(self):
        return self._input_dim