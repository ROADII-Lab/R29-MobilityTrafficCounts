import copy
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st
import altair as alt
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('module://drawilleplot')

class ai:

    # settings / object properties
    features = ['feature1', 'feature2', 'feature3']             # replace these at run time with the feature list from the data object!
    target = "target_feature1"                                  # this is what we are predicting, also supplied via the data object
    training_split = 0.05                                       # controls the amount of data to use for train/test
    model = None                                                # placeholder for the model once it has been initialized
    model_top = None                                            # placeholder for the top scoring model
    model_top_loss = 100                                        # placeholding for the current top model's test loss
    model_filename_root = "../models/model_"                    # default model filename
    model_filename = None                                       # placeholder for model filename
    model_size = 60                                             # number of parameters for the hidden network layer
    training_epochs = 2000                                      # default number of epochs to train the network for
    weight_decay = 0.001                                        # optimizer weight decay        
    dropout = 0.15                                              # % of neurons to apply dropout to                                        
    target_loss = 100                                           # keep training until either the epoch limit is hit or test loss is lower than this number
    training_learning_rate = 0.025                              # default network learning rate
    test_interval = 100                                         # model testing interval during training
    pdiffGoal = 0.15                        

    def __init__(self) -> None:
        
        # setup GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using ", self.device)

    def get_model_list(self, path, extension='pkl'):
        # returns a list of models in the specified path
        matching_files = []
        if not extension.startswith('.'):
            extension = '.' + extension
        
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(extension):
                    matching_files.append(os.path.join(root, file))
        
        return matching_files

    def format_training_data(self, dataframe):
        # formats features for input into the model
        # NOTE: this is not where the primary data features are created, that is performed within the data module itself.
        X = dataframe[self.features]
        y = dataframe[self.target]

        # data sanity checks
        if X.isnull().values.any():
            print("WARNING: Null values detected in training data!")
        
        if np.isinf(X).values.any():
            print("WARNING: Infinate values detected!")

        if X.duplicated().any():
            print(f'Duplicates: {len(X[X.duplicated()])}')
            print("WARNING: Duplicate rows detected!")

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.training_split, random_state=42,)

        # Output some basic debug info
        print("Training set size is", len(X_train),"records.")
        print("Test set size is", len(y_test),"records.")

        # Standardize the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Convert the data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train.astype(np.float32)).to(self.device)
        y_train_tensor = torch.tensor(y_train.values.astype(np.float32)).view(-1, 1).to(self.device)
        X_test_tensor = torch.tensor(X_test.astype(np.float32)).to(self.device)
        y_test_tensor = torch.tensor(y_test.values.astype(np.float32)).view(-1, 1).to(self.device)

        return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor
    
    def model_init(self, x_dim):
        # Initialize the neural network
        input_size = x_dim.shape[1]
        self.model = LinearNN(input_size, self.model_size, self.dropout)
        self.model.to(self.device)
        return 
    
    def model_load(self, x_dim, filename=model_filename):
        # loads model weights using specified filename
        self.model_init(x_dim)
        print("Loading model:", self.model_filename)
        if self.model.load(self.model_filename):
            self.model.to(self.device)
            return True
        else:
            return False
    
    def model_save(self, model):
        # triggers the model save process
        model.save(self.model_filename_root)
        self.model_filename = self.model_filename_root + ".pkl"

    def calculate_accuracy(self, predicted, known):
        if predicted.shape != known.shape:
            raise ValueError("The two tensors must be of the same shape!")

        print(known.shape[0])

        SST = torch.sum(torch.pow(known - torch.mean(known), 2))
        SSR = torch.sum(torch.pow((known - predicted), 2))

        percDiff = torch.divide(torch.abs(torch.sub(known, predicted)), known)

        # Move tensors to CPU for matplotlib operations
        percDiff_cpu = percDiff.cpu()

        plt.figure()
        bins = [i for i in range(0, max(100, int(100*torch.max(percDiff[percDiff.isfinite()]))+5), 5)]
        plt.hist(100*percDiff[percDiff.isfinite()], bins=[i for i in range(0, 155, 5)])
        plt.title('Distribution of Percent Difference between Expected and Predicted')
        plt.show()
        plt.close()

        numBelow10 = percDiff[percDiff < self.pdiffGoal]
        percBelow = 100 * numBelow10.shape[0] / known.shape[0]

        return 1 - SSR / SST, percBelow
    
    def plot_convergence(self, predicted, known):
        # Ensure tensors are moved to CPU before plotting
        predicted_cpu = predicted.cpu()
        known_cpu = known.cpu()

        plt.figure()
        plt.plot(known_cpu, predicted_cpu, 'k.')
        plt.ylim(top=int(torch.max(known_cpu) + (0.10 * torch.max(known_cpu))))
        plt.show()
        plt.close()

    def train(self, model, x_train, y_train, x_test, y_test, epochs = training_epochs, learning_rate=training_learning_rate):
        # model sanity checks
        with torch.no_grad():
            initial_outputs = model(x_train)
            if torch.isnan(initial_outputs).any():
                print('Initial outputs contain NaNs:', initial_outputs)

                for name, param in model.named_parameters():
                    if torch.isnan(param).any():
                        print(f'NaN found in {name} during initialization')
                
                test_input = torch.zeros_like(x_train)
                with torch.no_grad():
                    test_output = model(test_input)
                    if torch.isnan(test_output).any():
                        print('NaN produced with basic input! Check network initialization or definition!')
                
                # check for NaNs in the input data
                if torch.isnan(x_train).any():
                    print('NaN in x_train!')
                    nan_indices = torch.isnan(x_train)
                    if nan_indices.any():
                        print(f"NaN values found at: {nan_indices.nonzero()} in x_train!")
                if torch.isnan(y_train).any():
                    print('NaN in y_train')
                    nan_indices = torch.isnan(y_train)
                    if nan_indices.any():
                        print(f"NaN values found at: {nan_indices.nonzero()} in y_train!")
        
        # setup optimizer
        # Define the loss function and optimizer
        criterion = nn.HuberLoss(delta=500)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=self.weight_decay)
        
        # Place chart plotting epochs in a streamlit window
        col3, col4 = st.columns([1,1])
        with col4:
            pass
        with col3:
            # Initialize Plotting of Epoch and Loss
            data = pd.DataFrame({'Epoch': [], 'Loss':[]})
            chart = alt.Chart(data).encode(x=alt.X('Epoch', scale=alt.Scale(domain=(0,epochs)), axis=alt.Axis(title='Epochs')),
                                            y=alt.Y('Loss',scale=alt.Scale(type='log'), axis=alt.Axis(title='Logarithmic Loss'))).mark_line(color='red')
            chart = chart.properties(title='Logarithmic Loss vs Epochs')
            alt_chart = st.altair_chart(chart, use_container_width=False)
            
            # train the model
            for epoch in range(epochs):
                model.train()
                outputs = model(x_train)
                loss = criterion(outputs, y_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Create new data for this loop iteration
                new_data = pd.DataFrame({'Epoch': [epoch], 'Loss' : [loss.item()]})
                if epoch == 1:
                    data = new_data
                else:
                    data = pd.concat([data, new_data], ignore_index=True) 

                # create chart and add to existing container in streamlit
                chart = alt.Chart(data).encode(x=alt.X('Epoch', scale=alt.Scale(domain=(0,epochs)), axis=alt.Axis(title='Epochs')), 
                                            y=alt.Y('Loss',scale=alt.Scale(type='log'), axis=alt.Axis(title='Logarithmic Loss'))).mark_line(color='red')
                alt_chart.altair_chart(chart)
                if (epoch+1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')
                if (epoch+1) % self.test_interval == 0:
                    predictions, y_test, test_loss, test_accuracy = self.test(model, x_test, y_test)
                    self.plot_convergence(predictions, y_test)
                    #print(f'  Test Loss: {test_loss}; Test Accuracy: {test_accuracy}')

                    # if the loss is less, copy the weights, if we have hit the target loss, save the model and end training
                    if epoch+1 == self.test_interval:
                        self.model_top_loss = test_loss
                        self.model_top = copy.deepcopy(self.model)
                    if test_loss < self.model_top_loss:
                        self.model_top = copy.deepcopy(self.model)
                        self.model_top_loss = test_loss
                        if test_loss <= self.target_loss:
                            print("Early stopping!")
                            self.model_save(self.model_top)
                            return
        
        # save and make sure we set the model to the best weights
        self.model_save(self.model_top)
        self.model = self.model_top

    def test(self, model, x_test, y_test):
        # setup loss function
        criterion = nn.MSELoss()
        
        # Evaluate the model
        model.eval()
        with torch.no_grad():
            predictions = model(x_test)
            print(predictions)
            print(y_test)
            R2, Within10 = self.calculate_accuracy(predictions, y_test)
            test_loss = criterion(predictions, y_test)
            print(f'Test Loss: {test_loss.item()}, R2: {R2}, {Within10}% are within {100*self.pdiffGoal} Percent of Expected')
            return predictions, y_test, test_loss.item(), (R2, Within10)

    def predict(self, model, data):
        scaler = StandardScaler()
        data = scaler.transform(data[self.features])
        segments_without_counts_tensor = torch.tensor(data.astype(np.float32))
        with torch.no_grad():
            predicted_counts = model(segments_without_counts_tensor)
            return predicted_counts

# *Somewhat* simple neural network!
class LinearNN(nn.Module):
    def __init__(self, input_size, layer_size, dropout):
        super(LinearNN, self).__init__()
        self.fc1 = nn.Linear(input_size, int(layer_size*2))
        self.bn1 = nn.BatchNorm1d(int(layer_size*2))
        self.fc2 = nn.Linear(int(layer_size*2), layer_size)
        self.bn2 = nn.BatchNorm1d(layer_size)
        self.fc3 = nn.Linear(int(layer_size), layer_size)
        self.bn3 = nn.BatchNorm1d(layer_size)
        self.fc4 = nn.Linear(layer_size, layer_size // 2)  # Reduce layer size
        self.bn4 = nn.BatchNorm1d(layer_size // 2)
        self.fc5 = nn.Linear(layer_size // 2, 1)  # Output layer for regression

        # Optional: add dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        # x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.fc4(x)))
        # x = self.dropout(x)
        x = self.fc5(x)  # No activation function here as it's a regression task

        return x
    
    # Function to save the model
    def save(self, filename):
        # just save the model weights
        filename = self.create_filename(filename)
        torch.save(self.state_dict(), filename + ".pkl")
        print(f"Model weights saved to {filename}")


        # save the entire model for stand-alone inference later
        model_jit = torch.jit.script(self)
        model_jit.save(filename + ".pt")
        print(f"Model file saved to {filename}")

        return True

    # Function to load the model
    def load(self, filename):
        try:
            self.load_state_dict(torch.load(filename))
            self.eval()
            print(f"Model loaded from {filename}")
            return True
        except Exception as e:
            print(e)
            return False
    
    def create_filename(self, filename):
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{filename}_{current_datetime}"
