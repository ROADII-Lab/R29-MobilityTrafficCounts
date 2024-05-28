import warnings
import logging

# Suppress all warnings
warnings.filterwarnings("ignore")

# Suppress specific Streamlit warnings by setting the logging level to ERROR
logging.getLogger('streamlit').setLevel(logging.ERROR)

import copy
import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import altair as alt
import streamlit as st
import matplotlib
matplotlib.use('module://drawilleplot')

class ai:
    # settings / object properties
    features = ['feature1', 'feature2', 'feature3']
    target = "target_feature1"
    training_split = 0.05
    model = None
    model_top = None
    model_top_loss = 100
    model_filename_root = "../models/model_"
    model_filename = None
    model_size = 1000
    train_loader = None
    test_loader = None
    training_epochs = 100
    training_batch_size = 100000
    training_workers = 16
    testing_workers = 4
    weight_decay = 0.001
    dropout = 0.15
    target_loss = 100
    training_learning_rate = 0.05
    test_interval = 2
    pdiffGoal = 0.15

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if(self.device == "cuda"):
            torch.cuda.init()
            print("Using ", self.device)
        self.feature_importance_df = pd.DataFrame(columns=['epoch'] + self.features)

    def get_model_list(self, path, extension='pt'):
        matching_files = []
        if not extension.startswith('.'):
            extension = '.' + extension
        
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(extension):
                    matching_files.append(os.path.join(root, file))
        
        return matching_files
    
    def set_max_batch_size(self, x_train, y_train):
        print("Determining optimal batch size...")
        model = self.model.to(self.device)
        batch_size = self.training_batch_size
        max_memory_used = 0
        acceptable_memory = torch.cuda.get_device_properties(self.device).total_memory * 0.8

        while True:
            try:
                train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
                print("Testing batch size:", batch_size)

                for x_batch, y_batch in train_loader:
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                    with torch.no_grad():
                        output = model(x_batch)
                    break

                memory_used = torch.cuda.memory_allocated(self.device)
                if memory_used < acceptable_memory and memory_used > max_memory_used:
                    max_memory_used = memory_used
                    batch_size *= 2
                else:
                    break
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print("CUDA out of memory with batch size:", batch_size)
                    batch_size //= 2
                    if batch_size < 1:
                        break
                else:
                    raise e

        model = None
        torch.cuda.empty_cache()
        self.training_batch_size = batch_size

    def format_training_data(self, dataframe):
        X = dataframe[self.features]
        y = dataframe[self.target]

        if X.isnull().values.any():
            print("WARNING: Null values detected in training data!")
        
        if np.isinf(X).values.any():
            print("WARNING: Infinate values detected!")

        if X.duplicated().any():
            print(f'Duplicates: {len(X[X.duplicated()])}')
            print("WARNING: Duplicate rows detected!")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.training_split, random_state=42)

        print("Training set size is", len(X_train),"records.")
        print("Test set size is", len(y_test),"records.")

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train_tensor = torch.tensor(X_train.astype(np.float32))
        y_train_tensor = torch.tensor(y_train.values.astype(np.float32)).view(-1, 1)
        X_test_tensor = torch.tensor(X_test.astype(np.float32))
        y_test_tensor = torch.tensor(y_test.values.astype(np.float32)).view(-1, 1)

        return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor
    
    def model_init(self, x_dim):
        input_size = x_dim.shape[1]
        self.model = LinearNN(input_size, self.model_size, self.dropout)
        self.model.to(self.device)
        return 
    
    def model_load(self, x_dim, filename=model_filename):
        self.model_init(x_dim)
        print("Loading model:", self.model_filename)
        if self.model.load_model_for_inference(self.model_filename):
            self.model.to(self.device)
            return True
        else:
            return False
    
    def model_save(self, model):
        model.save(self.model_filename_root)
        self.model_filename = self.model_filename_root + ".pkl"

    def calculate_accuracy(self, predicted, known):
        predicted_cpu = predicted.cpu()
        known_cpu = known.cpu()

        if predicted_cpu.shape != known_cpu.shape:
            raise ValueError("The two tensors must be of the same shape!")

        print(known_cpu.shape[0])

        SST = torch.sum(torch.pow(known_cpu - torch.mean(known_cpu), 2))
        SSR = torch.sum(torch.pow((known_cpu - predicted_cpu), 2))

        percDiff = torch.divide(torch.abs(torch.sub(known_cpu, predicted_cpu)), known_cpu)

        plt.figure()
        plt.hist(100*percDiff[percDiff.isfinite()], bins=[i for i in range(0, 155, 5)])
        plt.title('Distribution of Percent Difference between Expected and Predicted')
        plt.show()
        plt.close()

        numBelow10 = percDiff[percDiff < self.pdiffGoal]
        percBelow = 100 * numBelow10.shape[0] / known.shape[0]

        return 1 - SSR / SST, percBelow
    
    def plot_convergence(self, predicted, known):
        predicted_cpu = predicted.cpu()
        known_cpu = known.cpu()

        plt.figure()
        plt.plot(known_cpu, predicted_cpu, 'k.')
        plt.ylim(top=int(torch.max(known_cpu) + (0.10 * torch.max(known_cpu))))
        plt.show()
        plt.close()

    def train(self, model, x_train, y_train, x_test, y_test, epochs=training_epochs, learning_rate=training_learning_rate):
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            torch.cuda.synchronize()
            model = nn.DataParallel(model)

        model.to(self.device)

        train_dataset = TensorDataset(x_train, y_train)
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.training_batch_size, shuffle=True, num_workers=self.training_workers, pin_memory=True)
        test_dataset = TensorDataset(x_test, y_test)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=self.training_batch_size, shuffle=False, num_workers=self.testing_workers, pin_memory=True)

        criterion = nn.HuberLoss(delta=500)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=self.weight_decay)

        col3, col4 = st.columns([1, 1])
        with col4:
            pass
        with col3:
            data = pd.DataFrame({'Epoch': [], 'Loss': []})
            chart = alt.Chart(data).mark_line(color='red').encode(
                x=alt.X('Epoch', scale=alt.Scale(domain=(0, epochs)), axis=alt.Axis(title='Epochs')),
                y=alt.Y('Loss', scale=alt.Scale(type='log'), axis=alt.Axis(title='Logarithmic Loss'))
            ).properties(title='Logarithmic Loss vs Epochs')
            alt_chart = st.altair_chart(chart, use_container_width=False)

            feature_importance_chart = None

            for epoch in range(epochs):
                epoch_start = time.time()
                model.train()
                total_loss = 0
                for x_batch, y_batch in self.train_loader:
                    x_batch, y_batch = x_batch.to(self.device, non_blocking=True), y_batch.to(self.device, non_blocking=True)
                    outputs = model(x_batch)
                    loss = criterion(outputs, y_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                average_loss = total_loss / len(self.train_loader)
                epoch_duration = time.time() - epoch_start

                new_data = pd.DataFrame({'Epoch': [epoch], 'Loss': [average_loss]})
                data = pd.concat([data, new_data], ignore_index=True)
                chart = alt.Chart(data).mark_line(color='red').encode(
                    x=alt.X('Epoch', scale=alt.Scale(domain=(0, epochs)), axis=alt.Axis(title='Epochs')),
                    y=alt.Y('Loss', scale=alt.Scale(type='log'), axis=alt.Axis(title='Logarithmic Loss'))
                )
                alt_chart.altair_chart(chart)

                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {average_loss:.4f}, Duration: {epoch_duration:.2f}s')

                if (epoch + 1) % self.test_interval == 0:
                    predictions, y_test, test_loss, test_accuracy = self.test(model, self.test_loader)
                    self.plot_convergence(predictions, y_test)
                    print(f'Test Loss: {test_loss:.4f}; Test Accuracy: {test_accuracy[0]:.4f}; {test_accuracy[1]:.2f}% are within {100*self.pdiffGoal}% of expected')

                    # Save the top model if the test loss is improved
                    if test_loss < self.model_top_loss:
                        self.model_top = copy.deepcopy(model.module if isinstance(model, nn.DataParallel) else model)
                        self.model_top_loss = test_loss
                        self.model_save(self.model_top)
                        if test_loss <= self.target_loss:
                            print("Early stopping!")
                            return

                    # Calculate feature importance
                    feature_importance = self.calculate_feature_importance(model, x_test, y_test)
                    feature_importance['epoch'] = epoch + 1
                    self.feature_importance_df = pd.concat([self.feature_importance_df, feature_importance], ignore_index=True)

                    # Visualize feature importance
                    melted_df = self.feature_importance_df.melt(id_vars=['epoch'], var_name='Feature', value_name='Importance')
                    melted_df['Importance'] = pd.to_numeric(melted_df['Importance'], errors='coerce')
                    top_features = melted_df.groupby('Feature')['Importance'].mean().nlargest(10).index.tolist()
                    melted_df = melted_df[melted_df['Feature'].isin(top_features)]
                    importance_chart = alt.Chart(melted_df).mark_line().encode(
                        x=alt.X('epoch:Q', axis=alt.Axis(title='Epochs')),
                        y=alt.Y('Importance:Q', scale=alt.Scale(type='log'), axis=alt.Axis(title='Average Change in Mean Square Error')),
                        color='Feature:N'
                    ).properties(title='Top 10: Feature Importance')
                    if feature_importance_chart is None:
                        feature_importance_chart = st.altair_chart(importance_chart, use_container_width=True)
                    else:
                        feature_importance_chart.altair_chart(importance_chart)

            self.model_save(self.model_top)
            self.model = self.model_top

    def calculate_feature_importance(self, model, x_test, y_test):
        model.eval()
        baseline_predictions = model(x_test.to(self.device)).cpu().detach().numpy()
        baseline_mse = np.mean((y_test.numpy() - baseline_predictions) ** 2)
        feature_importance = {}

        for i, feature in enumerate(self.features):
            x_test_permuted = x_test.clone()
            x_test_permuted[:, i] = x_test_permuted[:, i][torch.randperm(x_test_permuted.size()[0])]
            permuted_predictions = model(x_test_permuted.to(self.device)).cpu().detach().numpy()
            permuted_mse = np.mean((y_test.numpy() - permuted_predictions) ** 2)
            feature_importance[feature] = permuted_mse - baseline_mse

        return pd.DataFrame([feature_importance])

    def test(self, model, test_loader):
        model.eval()
        with torch.no_grad():
            total_loss = 0
            all_predictions = []
            all_y_test = []

            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(self.device, non_blocking=True), y_batch.to(self.device, non_blocking=True)
                predictions = model(x_batch)
                loss = F.mse_loss(predictions, y_batch)
                total_loss += loss.item()

                all_predictions.append(predictions.cpu())
                all_y_test.append(y_batch.cpu())

            all_predictions = torch.cat(all_predictions, dim=0)
            all_y_test = torch.cat(all_y_test, dim=0)

            R2, percBelow = self.calculate_accuracy(all_predictions, all_y_test)
            average_test_loss = total_loss / len(test_loader)

            print(f'Test Loss: {average_test_loss}, R2: {R2}, {percBelow:.2f}% are within {100 * self.pdiffGoal}% of Expected')

            return all_predictions, all_y_test, average_test_loss, (R2, percBelow)

class LinearNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.0):
        super(LinearNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def save(self, filename):
        torch.save(self.state_dict(), filename)
        print(f'Model saved to {filename}')

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
        print(f'Model loaded from {filename}')

    def save_model_for_inference(self, filename):
        torch.save(self.state_dict(), filename)
        print(f'Model for inference saved to {filename}')

    def load_model_for_inference(self, filename):
        self.load_state_dict(torch.load(filename))
        print(f'Model for inference loaded from {filename}')
        return True
