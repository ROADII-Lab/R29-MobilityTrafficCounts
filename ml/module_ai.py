import copy
import os
import json
import time
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, BatchSampler, RandomSampler
import pandas as pd
import altair as alt
import streamlit as st
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st
import altair as alt
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('module://drawilleplot')

class ai:                   

    def __init__(self) -> None:
        
        # settings / object property defaults
        self.features = ['feature1', 'feature2', 'feature3']             # replace these at run time with the feature list from the data object!
        self.target = "target_feature1"                                  # this is what we are predicting, also supplied via the data object
        self.training_split = 0.10                                       # controls the amount of data to use for train/test
        self.model = None                                                # placeholder for the model once it has been initialized
        self.model_version = 1                                           # number of times model is replaced during the training process due to better performance
        self.model_top = None                                            # placeholder for the top scoring model
        self.model_top_loss = 0                                          # placeholder for the current top model's test loss
        self.model_R2 = 0                                                # placeholder for the model's current R2 score
        self.model_within_percent = 0                                    # placeholder for the model's current within x% score
        self.model_filename_root = "../models/model_"                    # default model filename
        self.model_size = 600                                            # number of parameters for the hidden network layer
        self.train_loader = None                                         # placeholder for the training dataloader
        self.test_loader = None                                          # placeholder for the test dataloader
        self.training_epochs = 1500                                      # default number of epochs to train the network for
        self.current_training_epoch = 0                                  # placeholder for the current epoch counter
        self.training_batch_size = 850000                                # number of records we *think* we can fit into the GPU...
        self.test_batch_size = 850000                                    # number of records we *think* we can fit into the GPU...should be the same as above unless running into GPU memory issues
        self.training_workers = 16                                       # number of dataloader workers to use for loading training data into the GPU
        self.testing_workers = 4                                         # numer of dataloader workers to use for loading test data into the GPU
        self.weight_decay = 0.001                                        # optimizer weight decay        
        self.dropout = 0.15                                              # % of neurons to apply dropout to                                        
        self.training_learning_rate = 0.03                               # default network learning rate
        self.test_interval = 10                                         # model testing interval during training
        self.pdiffGoal = 0.15
        self.scaler = None                                               # placeholder for the scaler which is fitted to the training data     
        
        # setup GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda":
            torch.cuda.init()
            print("Using ", self.device)
        self.feature_importance_df = pd.DataFrame(columns=['epoch'] + self.features)  # Initialize DataFrame for feature importance

    def save_params_to_json(self, file_name='object_data.json'):
        def is_json_serializable(value):
            try:
                json.dumps(value)
                return True
            except (TypeError, OverflowError):
                return False
        
        def serialize(obj):
            if is_json_serializable(obj):
                return obj
            elif isinstance(obj, ai):
                return {key: self.serialize(value) for key, value in obj.__dict__.items()}
            elif isinstance(obj, dict):
                return {key: self.serialize(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [self.serialize(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(self.serialize(item) for item in obj)
            elif isinstance(obj, set):
                return {self.serialize(item) for item in obj}
            else:
                return str(obj)
            
        data = {key: serialize(value) for key, value in self.__dict__.items()}
        with open(file_name, 'w') as json_file:
            json.dump(data, json_file, indent=4)

    def get_model_list(self, path, extension='pt'): # def get_model_list(self, path, extension='pkl'):
        # returns a list of models in the specified path
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
                # Create DataLoader with the current batch size for each iteration
                train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
                
                print("Testing batch size:", batch_size)

                for x_batch, y_batch in train_loader:
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                    with torch.no_grad():
                        output = model(x_batch)
                    break # stop after one pass

                memory_used = torch.cuda.memory_allocated(self.device)
                if memory_used < acceptable_memory and memory_used > max_memory_used:
                    max_memory_used = memory_used
                    batch_size *= 2  # Increase batch size
                else:
                    break  # If memory exceeds acceptable limit or no more improvement, stop increasing
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print("CUDA out of memory with batch size:", batch_size)
                    batch_size //= 2  # Halve the batch size if out of memory
                    if batch_size < 1:
                        break
                else:
                    raise e

        model = None
        torch.cuda.empty_cache()  # Clear memory cache
        self.training_batch_size = batch_size

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
        print("Input Features:", list(X_train))

        # Output some basic debug info
        print("Training set size is", len(X_train),"records.")
        print("Test set size is", len(y_test),"records.")

        # Standardize the features
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # Convert the data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train.astype(np.float32))
        y_train_tensor = torch.tensor(y_train.values.astype(np.float32)).view(-1, 1)
        X_test_tensor = torch.tensor(X_test.astype(np.float32))
        y_test_tensor = torch.tensor(y_test.values.astype(np.float32)).view(-1, 1)

        return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

    def format_inference_data(self, dataframe):
        # formats features for input into the model to make predictions
        # NOTE: this is not where the primary data features are created, that is performed within the data module itself.
        X = dataframe[self.features]

        # data sanity checks
        if X.isnull().values.any():
            print("WARNING: Null values detected in training data!")
        
        if np.isinf(X).values.any():
            print("WARNING: Infinate values detected!")

        if X.duplicated().any():
            print(f'Duplicates: {len(X[X.duplicated()])}')
            print("WARNING: Duplicate rows detected!")

        print("Input Features:", list(X))

        # Output some basic debug info
        print("Data set size is", len(X),"records.")

        # Standardize the features using the same scaler fitted during training (loaded from disk)
        X = self.scaler.transform(X)

        return torch.tensor(X.astype(np.float32))
    
    def model_inference(self, dataframe):
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            torch.cuda.synchronize()
            self.model = nn.DataParallel(self.model)
        
        # Ensure the model is in evaluation mode
        self.model.eval()
        
        # Format the inference data
        X_tensor = self.format_inference_data(dataframe)
        
        # Move data to the appropriate data structure for batch processing
        inference_dataset = TensorDataset(X_tensor)
        inference_loader = DataLoader(dataset=inference_dataset, batch_size=self.test_batch_size, shuffle=False, num_workers=self.testing_workers, pin_memory=True)
        
        # Initialize an empty list to store predictions
        all_predictions = []
        
        # Perform inference in batches
        with torch.no_grad():
            for batch_X_tensor in inference_loader:
                batch_X_tensor = batch_X_tensor[0].to(self.device)  # Assuming model and data are on the same device
                batch_predictions = self.model(batch_X_tensor)
                all_predictions.append(batch_predictions)
        
        # Concatenate all predictions
        # predictions = np.concatenate(all_predictions, axis=0)
        predictions = torch.cat(all_predictions, dim=0).cpu()

        # Convert predictions to integers
        df = pd.DataFrame(predictions, columns=['predicted'])
        df['predicted'] = df['predicted'].astype(int)
        
        return df
    
    def scaler_load(self, filename):
        # load the previously fitted scaler for inference
        self.scaler = StandardScaler()
        self.scaler = joblib.load(filename)

        return 

    def scaler_save(self, scaler, filename):
        # saves the mean standard diviation scaler to a pickle file to be used later during inference
        joblib.dump(scaler, filename)

    def model_init(self, x_dim, filename=None):
        # Initialize the neural network
        input_size = x_dim.shape[1]
        self.model = LinearNN(input_size, self.model_size, self.dropout, self.model_filename_root, filename=filename)
        self.model.to(self.device)
        return 

    def model_load(self, dataframe, filename):
        # loads the model using specified filename
        self.model_init(dataframe, filename=filename)
        # Extract the filename plus path root from self.filename and update self.filename_root
        self.model.filename_root = os.path.splitext(filename)[0]
        self.model.filename = filename
        self.model.to(self.device)
        self.scaler_load(self.model.filename_root + "_scaler.pkl")

    def model_save(self, specific_model):
        # triggers the model save process, saving the model plus any assocaited parameters as a .pt and .json files
        specific_model.save()
        self.scaler_save(self.scaler, specific_model.filename_root + "_scaler.pkl")
        self.save_params_to_json(specific_model.filename_root + "_params.json")

    def calculate_accuracy(self, predicted, known):

        if predicted.shape != known.shape:
            raise ValueError("The two tensors must be of the same shape!")

        print("Test set size =", known.shape[0])

        SST = torch.sum(torch.pow(known - torch.mean(known), 2))
        SSR = torch.sum(torch.pow((known - predicted), 2))

        percDiff = torch.divide(torch.abs(torch.sub(known, predicted)), known)

        plt.figure()
        plt.hist(100*percDiff[percDiff.isfinite()], bins=[i for i in range(0, 155, 5)])
        plt.title('Distribution of Percent Difference between Expected and Predicted')
        plt.show()
        plt.close()

        numBelow10 = percDiff[percDiff < self.pdiffGoal]
        percBelow = 100 * numBelow10.shape[0] / known.shape[0]

        return 1 - SSR / SST, percBelow
    
    def plot_convergence(self, predicted, known):

        plt.figure()
        plt.plot(known, predicted, 'k.')
        plt.ylim(top=int(torch.max(known) + (0.10 * torch.max(known))))
        plt.show()
        plt.close()

    def train(self, model, x_train, y_train, x_test, y_test, accumulation_steps=4):
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            torch.cuda.synchronize()
            model = nn.DataParallel(model)

        model.to(self.device)

        train_dataset = TensorDataset(x_train, y_train)
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.training_batch_size // accumulation_steps, shuffle=True, num_workers=self.training_workers, pin_memory=True)
        test_dataset = TensorDataset(x_test, y_test)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=self.test_batch_size, shuffle=False, num_workers=self.testing_workers, pin_memory=True)

        criterion = nn.HuberLoss(delta=500)
        optimizer = optim.AdamW(model.parameters(), lr=self.training_learning_rate, weight_decay=self.weight_decay)

        col3, col4 = st.columns([1, 1])
        with col4:
            pass
        with col3:
            data = pd.DataFrame({'Epoch': [], 'Loss': []})
            chart = alt.Chart(data).mark_line(color='red').encode(
                x=alt.X('Epoch', scale=alt.Scale(domain=(0, self.training_epochs)), axis=alt.Axis(title='Epochs')),
                y=alt.Y('Loss', scale=alt.Scale(type='log'), axis=alt.Axis(title='Logarithmic Loss'))
            ).properties(title='Logarithmic Loss vs Epochs')
            alt_chart = st.altair_chart(chart, use_container_width=False)

            feature_importance_chart = st.empty()  # Initialize an empty container for the feature importance chart

            for epoch in range(self.training_epochs):
                self.current_training_epoch = (epoch+1)
                epoch_start = time.time()  # Start time of the epoch
                model.train()
                total_loss = 0
                optimizer.zero_grad()

                for i, (x_batch, y_batch) in enumerate(self.train_loader):
                    x_batch, y_batch = x_batch.to(self.device, non_blocking=True), y_batch.to(self.device, non_blocking=True)
                    # Forward pass
                    outputs = model(x_batch)
                    loss = criterion(outputs, y_batch)
                    # Backward pass and optimization
                    loss.backward()

                    if (i + 1) % accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                    total_loss += loss.item()

                average_loss = total_loss / len(self.train_loader)
                epoch_duration = time.time() - epoch_start  # Calculate duration of the epoch

                # Update and display the chart
                new_data = pd.DataFrame({'Epoch': [epoch], 'Loss': [average_loss]})
                data = pd.concat([data, new_data], ignore_index=True)
                chart = alt.Chart(data).mark_line(color='red').encode(
                    x=alt.X('Epoch', scale=alt.Scale(domain=(0, self.training_epochs)), axis=alt.Axis(title='Epochs')), 
                    y=alt.Y('Loss', scale=alt.Scale(type='log'), axis=alt.Axis(title='Logarithmic Loss'))
                )
                alt_chart.altair_chart(chart)

                print(f'Epoch [{epoch+1}/{self.training_epochs}], Loss: {average_loss:.4f}, Time: {epoch_duration:.2f} sec')

                if (epoch+1) % self.test_interval == 0:
                    # Predictions and test metrics
                    model.eval()
                    # total_test_loss = 0
                    all_predictions = []
                    all_y_test = []

                    all_predictions, all_y_test, test_loss, R2, Within_percent = self.test(model, self.test_loader)

                    # plot data within x %
                    print(f'Test Loss: {test_loss:.4f}; R2: {R2:.4f}; {Within_percent:.2f}% are within {100*self.pdiffGoal}% of expected')
                    
                    # plot convergence
                    self.plot_convergence(all_predictions, all_y_test)
                    print(f'  Test Loss: {test_loss}; Test Accuracy: {R2}')

                    # Make sure we copy the model if none exists
                    if(self.model_top == None):
                        self.model_top = copy.deepcopy(model.module if isinstance(model, nn.DataParallel) else model)

                    # need to use our metrics to keep the highest performing model
                    if R2 >= self.model_R2 and Within_percent >= self.model_within_percent:
                        self.model_top = copy.deepcopy(model.module if isinstance(model, nn.DataParallel) else model)
                        self.model_top_loss = test_loss
                        self.model_R2 = float(R2)
                        self.model_within_percent = Within_percent
                        self.model_save(self.model_top)

                    # Calculate feature importance
                    feature_importance = self.calculate_feature_importance(model, x_test, y_test)
                    feature_importance['epoch'] = epoch + 1
                    self.feature_importance_df = pd.concat([self.feature_importance_df, feature_importance], ignore_index=True)

                    # Save feature importance to CSV
                    self.feature_importance_df.to_csv(self.model.filename_root + "_featureimportance.csv", index=False)

                    # Exclude columns with 'before' or 'after' in their names
                    filtered_df = self.feature_importance_df.loc[:, ~self.feature_importance_df.columns.str.contains('before|after')]

                    # Identify the largest epoch
                    largest_epoch = filtered_df['epoch'].max()

                    # Filter the dataframe for the largest epoch
                    largest_epoch_df = filtered_df[filtered_df['epoch'] == largest_epoch]

                    # Drop the 'epoch' column as it's not needed for feature importance
                    largest_epoch_df = largest_epoch_df.drop(columns=['epoch'])

                    # Transpose the dataframe to get features as rows
                    transposed_df = largest_epoch_df.T
                    transposed_df.columns = ['importance']
                    transposed_df['importance'] = pd.to_numeric(transposed_df['importance'])

                    # Sort by importance and get the top 10 features
                    top_10_features = transposed_df.nlargest(10, 'importance')

                    # Plotting feature importance
                    #self.plot_feature_importance_terminal(top_10_features)
                    self.plot_feature_importance_streamlit(top_10_features, feature_importance_chart)

            # Final model saving after training
            if(self.model_top != None):
                self.model = self.model_top
            self.model_save(self.model)

    def test(self, model, test_loader):
        # criterion = nn.MSELoss()
        criterion = nn.HuberLoss(delta=500)
        model.eval()

        total_loss = 0
        all_predictions = []
        all_y_test = []

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(self.device, non_blocking=True), y_batch.to(self.device, non_blocking=True)
                predictions = model(x_batch)
                test_loss = criterion(predictions, y_batch)
                total_loss += test_loss.item()

                all_predictions.append(predictions)
                all_y_test.append(y_batch)

        # Concatenate all batches for calculating accuracy and other metrics
        all_predictions = torch.cat(all_predictions, dim=0).cpu()
        all_y_test = torch.cat(all_y_test, dim=0).cpu()

        R2, Within10 = self.calculate_accuracy(all_predictions, all_y_test)
        average_test_loss = total_loss / len(test_loader)
        
        # Release GPU memory
        del x_batch, y_batch, predictions, test_loss
        torch.cuda.empty_cache()

        # Create DataFrame
        df = pd.DataFrame(all_y_test, columns=['actual'])
        df['predicted'] = all_predictions

        # force predictions to be integrers
        df['predicted'] = df['predicted'].astype(int)
        print(df)

        return df, all_y_test, average_test_loss, R2, Within10

    def plot_feature_importance_terminal(self, top_10_features):
        # Plotting with Matplotlib for terminal
        plt.figure(figsize=(12, 8))
        bars = plt.barh(top_10_features.index, top_10_features['importance'], color='skyblue')
        plt.xscale('log')
        plt.xlabel('Average Mean Square Error')
        plt.title('Top 10: Feature Importance')
        plt.gca().invert_yaxis()  # Invert y-axis to have the greatest feature on top

        # Adding labels
        for bar in bars:
            plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{bar.get_width():.2e}', va='center', ha='left')

        plt.show()


    def plot_feature_importance_streamlit(self, top_10_features, feature_importance_chart):
        # Plotting with Altair for Streamlit
        top_10_features_reset = top_10_features.reset_index()
        bar_chart = alt.Chart(top_10_features_reset).mark_bar().encode(
            y=alt.Y('index:N', sort='-x', title='Feature'),
            x=alt.X('importance:Q', scale=alt.Scale(type='log'), title='Average Mean Square Error'),
            tooltip=['index', 'importance']
        ).properties(
            title='Top 10: Feature Importance'
        )

        text = bar_chart.mark_text(
            align='left',
            baseline='middle',
            dx=3  # Nudges text to right so it doesn't appear on top of the bar
        ).encode(
            text=alt.Text('importance:Q', format='.2e')
        )

        combined_chart = (bar_chart + text).configure_axis(
            labelFontSize=12,
            titleFontSize=14
        ).configure_title(
            fontSize=16
        )

        feature_importance_chart.altair_chart(combined_chart, use_container_width=True)

    def calculate_feature_importance(self, model, x_test, y_test):
        model.eval()
        baseline_predictions = self._predict_in_batches(model, x_test)
        baseline_mse = np.mean((y_test.numpy() - baseline_predictions) ** 2)
        feature_importance = {}

        for i, feature in enumerate(self.features):
            x_test_permuted = x_test.clone()
            x_test_permuted[:, i] = x_test_permuted[:, i][torch.randperm(x_test_permuted.size()[0])]
            permuted_predictions = self._predict_in_batches(model, x_test_permuted)
            permuted_mse = np.mean((y_test.numpy() - permuted_predictions) ** 2)
            feature_importance[feature] = permuted_mse - baseline_mse

        return pd.DataFrame([feature_importance])

    def _predict_in_batches(self, model, x_test):
        predictions = []
        for i in range(0, x_test.size(0), self.test_batch_size):
            x_batch = x_test[i:i+self.test_batch_size].to(self.device)
            with torch.no_grad():
                batch_predictions = model(x_batch).cpu().numpy()
            predictions.append(batch_predictions)
        return np.concatenate(predictions)

    def predict(self, model, data):
        scaler = StandardScaler()
        data = scaler.transform(data[self.features])
        segments_without_counts_tensor = torch.tensor(data.astype(np.float32))
        with torch.no_grad():
            predicted_counts = model(segments_without_counts_tensor)
            return predicted_counts

# *Somewhat* simple neural network!
class LinearNN(nn.Module):

    def __init__(self, input_size, layer_size, dropout, filename_root, filename=None):
        super(LinearNN, self).__init__()
        self.fc1 = nn.Linear(input_size, int(layer_size*2))
        self.bn1 = nn.BatchNorm1d(int(layer_size*2))
        self.fc2 = nn.Linear(int(layer_size*2), layer_size)
        self.bn2 = nn.BatchNorm1d(layer_size)
        self.fc3 = nn.Linear(int(layer_size), layer_size)
        self.bn3 = nn.BatchNorm1d(layer_size)
        self.fc4 = nn.Linear(layer_size, layer_size // 2) 
        self.bn4 = nn.BatchNorm1d(layer_size // 2)
        self.fc5 = nn.Linear(layer_size // 2, 1)  # Output layer for regression

        # Optional: add dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # setup filename
        if filename == None:
            self.filename = self.create_filename(filename_root)
        else:
            self.filename = filename
            self.load_model_for_inference()

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = self.fc5(x)  # No activation function here as it's a regression task

        return x
    
    # Function to save the model
    def save(self):

        # save the entire model for stand-alone inference later
        model_jit = torch.jit.script(self)
        model_jit.save(self.filename)
        print(f"Model file saved to {self.filename}")

        return True
        
    def load_model_for_inference(self):
        try:
            # Load the entire JIT-compiled model
            loaded_model = torch.jit.load(self.filename, map_location=torch.device('cpu'))
            # Switch the model to evaluation mode
            loaded_model.eval()

            # Transfer the state from the loaded model to the current instance
            self.load_state_dict(loaded_model.state_dict())

            print(f"Model loaded from {self.filename}")

            return True

        except Exception as e:
            # Print the exception and return False if any error occurs
            print(e)
            return False
    
    def create_filename(self, filename_root):
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename_root = f"{filename_root}_{current_datetime}"
        return self.filename_root + ".pt"