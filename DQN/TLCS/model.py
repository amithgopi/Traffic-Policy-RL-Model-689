import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # kill warning about tensorflow
import tensorflow as tf
import numpy as np
import sys

import torch.nn as nn
import torch

from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras import losses
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

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
        for _ in range(num_layers):
            self.layers.append(nn.Linear(width, width))
        self.layers.append(nn.Linear(width, output_dim))
    
    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return x

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


    def train_batch(self, states, q_sa):
        """
        Train the nn using the updated q-values
        """
        states = torch.tensor(states).float().to(device)
        currentValues = self._model(states)
        targetValues = torch.tensor(q_sa).float().to(device)
        self.optimizer.zero_grad()
        # print(currentValues.size(), targetValues.size())
        loss = self.criterion(currentValues, targetValues)
        loss.backward()
        self.optimizer.step()
        # self._model.fit(states, q_sa, epochs=1, verbose=0)


    def save_model(self, path):
        """
        Save the current model in the folder as h5 file and a model architecture summary as png
        """
        torch.save(self._model, os.path.join(path, 'trained_model.h5'))
        # self._model.save(os.path.join(path, 'trained_model.h5'))
        # plot_model(self._model, to_file=os.path.join(path, 'model_structure.png'), show_shapes=True, show_layer_names=True)


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