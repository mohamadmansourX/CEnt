from typing import Callable, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from matplotlib import pyplot as plt
from carla.recourse_methods.processing import  merge_default_parameters
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Build the VAE
class VariationalAutoencoder(nn.Module):
    _DEAFULT_PARAMS = {
        'input_dim': 784,
        'layers': [512, 128],
        'latent_dim': 32,
        'hidden_activation': 'relu',
        'dropout': 0.2,
        'batch_norm': True,
        'batch_size': 64,
        'epochs': 30,
        'learning_rate': 0.001,
        'kld_weight': 0.0025,
        'weight_decay': 0.0,
        'cuda': False,
        'verbose': True,
        'train': True,
        'save_dir': './models',
    }
    def __init__(self, data_name: str, layers: List, mutable_mask, params = {}):
        super(VariationalAutoencoder, self).__init__()
        print(params)
        self.params = merge_default_parameters(params, self._DEAFULT_PARAMS)
        self.params['save_dir'] = os.path.join(self.params['save_dir'], data_name)
        print(self.params['save_dir'])
        # Define the encoder
        self._encoder, self._mu_enc, self._log_var_enc = self.define_encoder()
        # Define the decoder
        self._mu_dec = self.define_decoder()
        # Define the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.params['learning_rate'],
                                    weight_decay=self.params['weight_decay'])
        # Define the loss function
        self.criterion = nn.MSELoss(reduction='sum')
        # Define the cuda
        if self.params['cuda']:
            self.cuda()
        # Check if train is true
        if not self.params['train']:
            # Load the model from save_dir by searching for the path containing '_epoch_best_model.pth'
            # Search for the path containing '_epoch_best_model.pth'
            path = glob(os.path.join(self.params['save_dir'], '*_epoch_best_model.pth'))
            # Assert if path is empty list
            assert len(path) == 1, 'There should be only one file containing the best model, eith no checkpoint or more than one checkpoint'
            path = path[0]
            # Load the model
            self.load_model(path)
    
    def load_model(self, path):
        # Load the model
        self.load_state_dict(torch.load(path))
        # Print the model
        print('Model loaded from {}'.format(path))

    def forward(self, x):
        # Encode the input
        mu, log_var = self.encode(x)
        # Sample from the latent space
        z = self.sample(mu, log_var)
        # Decode the input
        x_hat = self.decode(z)
        # Return the mean and the log_var
        return x_hat, mu, log_var


    # Define plot loss
    def plot_loss(self, plot_flag = True, save_path = None):
        #self.loss_list = {'Steps':[],'Loss':[],
        #                  'Epoch_Loss':[],'Best_Epoch_Loss':[],
        #                  'Epochs':[]}
        
        # Plot Loss Vs Steps
        plt.figure(figsize=(10,5))
        plt.plot(self.loss_list['Steps'], self.loss_list['Loss'], label='Loss', color='red')
        # Set the x-axis label
        plt.xlabel('Steps')
        # Set the y-axis label
        plt.ylabel('Loss')
        # Set the title
        plt.title('Loss Vs Steps')
        # Set the legend
        plt.legend()
        # Show the plot
        if plot_flag:
            # Show the plot
            plt.show()
        # Plot Epoch Loss and Best Epoch Loss Vs Epochs
        plt.figure(figsize=(10,5))
        # Plot Epoch Loss in red
        plt.plot(self.loss_list['Epochs'], self.loss_list['Epoch_Loss'], label='Loss', color='red')
        # Plot Best Epoch Loss in blue
        plt.plot(self.loss_list['Epochs'], self.loss_list['Best_Epoch_Loss'], label='Best Loss', color='blue', linestyle='dashdot')
        # Plot Best Epoch Loss in blue
        plt.plot(self.loss_list['Epochs'], self.loss_list['Epoch_Test_MSE'], label='Test MSE Loss', color='green', linestyle='dashdot')
        # Set the x-axis label
        plt.xlabel('Epochs')
        # Set the y-axis label
        plt.ylabel('Loss')
        # Set the title
        plt.title('Loss Vs Epochs')
        # Set the legend
        plt.legend()
        if plot_flag:
            # Show the plot
            plt.show()
        else:
            # Save plt to plt.png
            plt.savefig(save_path)
    # Define training function
    def fit(self, xtrain: Union[pd.DataFrame, np.ndarray]
    ):
        # Create save_dir if it doesn't exist
        if not os.path.exists(self.params['save_dir']):
            os.makedirs(self.params['save_dir'])
        # Set the model to train mode
        self.train()
        # Define loss list for visualization
        self.loss_list = {'Steps':[],'Loss':[],
                          'Epoch_Loss':[],'Epoch_Test_MSE':[],'Best_Epoch_Loss':[],
                          'Epochs':[]}
        # Define Best loss param
        best_loss = -1
        best_checkpoint_path = None
        if isinstance(xtrain, pd.DataFrame):
            xtrain = xtrain.values
        
        # Split xtrain to train and test
        xtrain, xtest = train_test_split(xtrain, test_size=0.2, random_state=42)

        xtrain = torch.from_numpy(xtrain).type(torch.FloatTensor)
        xtest = torch.from_numpy(xtest).type(torch.FloatTensor)

        # loop on self.params['epochs']
        for epoch in range(self.params['epochs']):
            train_loader = torch.utils.data.DataLoader(
                xtrain, batch_size=self.params['batch_size'], shuffle=True
            )
            loss_epoch = []
            beta = epoch / self.params['epochs']
            # loop on the batches
            for i, (x) in enumerate(train_loader):
                if self.params['cuda']:
                    x = x.cuda()
                # Forward pass
                x_hat, mu, log_var = self.forward(x)
                # Compute the loss
                loss = self.loss_function(x, x_hat, mu, log_var, beta)
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_val = loss.data.detach().numpy()
                self.loss_list['Loss'].append(loss_val)
                steps = epoch * len(train_loader) + i
                self.loss_list['Steps'].append(steps)
                loss_epoch.append(loss_val)
            
            test_loader = torch.utils.data.DataLoader(
                xtest, batch_size=self.params['batch_size'], shuffle=True
            )
            loss_epoch_test = []
            # loop on the batches
            for i, (x) in enumerate(test_loader):
                if self.params['cuda']:
                    x = x.cuda()
                # Forward pass
                x_hat, mu, log_var = self.forward(x)
                # Compute the MSE loss
                loss_epoch_test.append(self.criterion(x_hat, x).detach().numpy())
            # Compute the mean loss
            loss_epoch_test = np.mean(loss_epoch_test)
            self.loss_list['Epoch_Test_MSE'].append(loss_epoch_test)

            print('Epoch: {}, ELBO Loss: {}, Test MSELoss: {}'.format(epoch, np.mean(loss_epoch), loss_epoch_test))
            if epoch == 0:
                # Set best_loss to loss_epoch mean
                best_loss = np.mean(loss_epoch)
                print('Epoch: {}, Best ELBO Loss: {}'.format(epoch, best_loss))
                best_checkpoint_path = os.path.join(self.params['save_dir'], '{}_epoch_best_model.pth'.format(epoch))
                # Save the model to the epoch_best_model.pth
                torch.save(self.state_dict(), best_checkpoint_path)
            else:
                # If loss_epoch mean is better than best_loss, set best_loss to loss_epoch mean
                if np.mean(loss_epoch) < best_loss:
                    best_loss = np.mean(loss_epoch)
                    print('BEST Epoch: {}, Best ELBO Loss: {}'.format(epoch, best_loss))
                    # Remove the previous best_checkpoint_path
                    os.remove(best_checkpoint_path)
                    # Save the model to the epoch_best_model.pth
                    best_checkpoint_path = os.path.join(self.params['save_dir'], '{}_epoch_best_model.pth'.format(epoch))
                    torch.save(self.state_dict(), best_checkpoint_path)
            self.loss_list['Epoch_Loss'].append(np.mean(loss_epoch))
            self.loss_list['Best_Epoch_Loss'].append(best_loss)
            self.loss_list['Epochs'].append(epoch)
        self.eval()    
    
    def loss_function(self, x, x_hat, mu, log_var,beta):
        # Compute the loss
        loss = self.criterion(x_hat, x)
        # Add the KL divergence
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(),
                                               dim = 1), dim = 0)
        # Return the loss
        loss = loss + beta * self.params['kld_weight'] * kld_loss
        return loss

    def get_encodings(self, x):
        if isinstance(x, pd.DataFrame):
            x = x.values
        if isinstance(x, list):
            x = np.array(x)
        x = torch.from_numpy(x).type(torch.FloatTensor)
        mu, log_var = self.encode(x)
        return self.sample(mu, log_var).detach().numpy()

    def sample(self, mu, log_var):
        # Sample from the latent space
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        x = self._encoder(x)
        # Encode the input
        h = self._mu_enc(x)
        # Return the mean and the log_var
        return h, self._log_var_enc(x)

    def decode(self, z):
        # Decode the input
        h = self._mu_dec(z)
        # Return the mean and the log_var
        return h

    def define_decoder(self):
        decoder_layers = []
        layers = self.params['layers'][::-1]
        for i in range(len(layers)):
            if i == 0:
                decoder_layers.append(nn.Linear(self.params['latent_dim'], layers[i]))
            else:
                decoder_layers.append(nn.Linear(layers[i-1], layers[i]))
            if self.params['batch_norm']:
                decoder_layers.append(nn.BatchNorm1d(layers[i]))
            # Check if activation name is valid in the torch.nn.functional
            if self.params['hidden_activation'].lower() == 'relu':
                decoder_layers.append(nn.ReLU())
            if self.params['dropout'] > 0:
                decoder_layers.append(nn.Dropout(self.params['dropout']))
                
        # Define the decoder
        decoder = nn.Sequential(*decoder_layers)
        # Define the mu
        mu_dec = nn.Linear(layers[-1], self.params['input_dim'])
        mu_dec = nn.Sequential(decoder, mu_dec)
        
        return mu_dec

    def define_encoder(self):
        encoder_layers = []
        for i in range(len(self.params['layers'])):
            if i == 0:
                encoder_layers.append(nn.Linear(self.params['input_dim'], self.params['layers'][i]))
            else:
                encoder_layers.append(nn.Linear(self.params['layers'][i-1], self.params['layers'][i]))
            if self.params['batch_norm']:
                encoder_layers.append(nn.BatchNorm1d(self.params['layers'][i]))
            # Check if activation name is valid in the torch.nn.functional
            if self.params['hidden_activation'].lower() == 'relu':
                encoder_layers.append(nn.ReLU())
            if self.params['dropout'] > 0:
                encoder_layers.append(nn.Dropout(self.params['dropout']))
        # Define the encoder
        encoder = nn.Sequential(*encoder_layers)
        # Define the mu
        mu_enc = nn.Linear(self.params['layers'][-1], self.params['latent_dim'])
        mu_enc = nn.Sequential(mu_enc)
        # Define the log_var
        log_var_enc = nn.Linear(self.params['layers'][-1], self.params['latent_dim'])
        log_var_enc = nn.Sequential(log_var_enc)
        return encoder, mu_enc, log_var_enc