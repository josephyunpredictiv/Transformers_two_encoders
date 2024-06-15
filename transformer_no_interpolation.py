# -*- coding: utf-8 -*-
"""
Original file is located at (Not avalible publicly due to containing classified info)
    https://colab.research.google.com/drive/1D5HcDgICwmPm_Pf221be_KUs1vFgDBz3
"""

"""# Transformers"""

# Python 3.10.12
"""##non interpolated"""

import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Masking, LayerNormalization, Dropout, Layer
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import psutil
import os
import time
import random
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
from sklearn.metrics import mean_squared_error
import pickle

# Function to set the seed for reproducibility
def set_seed(seed=37):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# Set the seed
set_seed(37)

# Function to get current RAM usage
def get_ram_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)  # Convert bytes to MB

def log_ram_usage(label):
    print(f"{label} - RAM usage: {get_ram_usage()} MB")

log_ram_usage("Before loading data")

file_path = 'XXXXXXXXXX'
data = pd.read_csv(file_path)

log_ram_usage("After loading data")

# Extract the numeric part from the ID_NXT column
data['ID_NXT'] = data['ID_NXT'].apply(lambda x: int(re.search(r'\d+', x).group()))

# List of columns to predict
target_columns = []  # Add as many as needed

# Sort by ID_NXT and APP_YEAR
data = data.sort_values(by=['ID_NXT', 'APP_YEAR'])

# Interpolate missing values within each group sorted by APP_YEAR
data[target_columns] = data.groupby('ID_NXT')[target_columns].apply(lambda group: group.interpolate(method='linear')).reset_index(level=0, drop=True)

# Normalize and prepare data
scaler = MinMaxScaler(feature_range=(0, 1))
# Save the scaler
scaler_path = 'scaler.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
data[target_columns] = scaler.fit_transform(data[target_columns])

log_ram_usage("After normalizing data")

# Group data by patient ID
grouped = data.groupby('ID_NXT')

# Prepare sequences for each patient
sequences = []
for name, group in grouped:
    sequences.append(group[target_columns].values)

log_ram_usage("After preparing sequences")

# Pad sequences to the same length
padded_sequences = pad_sequences(sequences, padding='post', dtype='float32')

log_ram_usage("After padding sequences")

# Prepare the data for the transformer model
def prepare_transformer_data(padded_sequences, time_step):
    X, y = [], []
    for sequence in padded_sequences:
        for i in range(len(sequence) - time_step):
            X.append(sequence[i:i + time_step])
            y.append(sequence[i + time_step])
    return np.array(X), np.array(y)

# Set the time step
time_step = 5

# Prepare the data with the specified time step
X, y = prepare_transformer_data(padded_sequences, time_step)

log_ram_usage("After preparing transformer data")

# Split the data into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training data dimensions")
print(X_train.shape)
print("Testing data dimensions")
print(X_test.shape)

log_ram_usage("After splitting data")


# Allows for masking
class CustomMultiHeadAttention(Layer):
    def __init__(self, head_size, num_heads, dropout=0.0, **kwargs):
        super(CustomMultiHeadAttention, self).__init__(**kwargs)
        self.head_size = head_size
        self.num_heads = num_heads
        self.dropout = dropout

    def build(self, input_shape):
        self.query_dense = Dense(self.head_size * self.num_heads)
        self.key_dense = Dense(self.head_size * self.num_heads)
        self.value_dense = Dense(self.head_size * self.num_heads)
        self.output_dense = Dense(input_shape[-1])
        self.dropout_layer = Dropout(self.dropout)

    def call(self, inputs, mask=None):
        query, key, value = inputs, inputs, inputs
        batch_size = tf.shape(query)[0]

        # Linear projections
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # Reshape to [batch_size, num_heads, seq_len, head_size]
        query = self._reshape_to_batches(query, batch_size)
        key = self._reshape_to_batches(key, batch_size)
        value = self._reshape_to_batches(value, batch_size)

        # Scaled dot-product attention
        attention_scores = tf.matmul(query, key, transpose_b=True)
        attention_scores = attention_scores / tf.sqrt(tf.cast(self.head_size, dtype=tf.float32))

        if mask is not None:
            mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype=tf.float32)
            attention_scores += (mask * -1e9)

        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_weights = self.dropout_layer(attention_weights)

        attention_output = tf.matmul(attention_weights, value)
        attention_output = self._reshape_from_batches(attention_output, batch_size)

        # Linear projection
        output = self.output_dense(attention_output)
        return output

    def compute_mask(self, inputs, mask=None):
        # Propagate the mask to downstream layers
        return mask

    def _reshape_to_batches(self, x, batch_size):
        x = tf.reshape(x, [batch_size, -1, self.num_heads, self.head_size])
        return tf.transpose(x, [0, 2, 1, 3])

    def _reshape_from_batches(self, x, batch_size):
        x = tf.transpose(x, [0, 2, 1, 3])
        return tf.reshape(x, [batch_size, -1, self.num_heads * self.head_size])

# Custom transformer block using CustomMultiHeadAttention which allows for masking
def transformer_block(inputs, head_size, num_heads, ff_dim, dropout=0):
    attention_output = CustomMultiHeadAttention(head_size, num_heads, dropout)(inputs)
    attention_output = Dropout(dropout)(attention_output)
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output + inputs)

    ff_output = Dense(ff_dim, activation='relu')(attention_output)
    ff_output = Dropout(dropout)(ff_output)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    ff_output = LayerNormalization(epsilon=1e-6)(ff_output + attention_output)

    return ff_output

# Custom layer for global average pooling
class GlobalAveragePooling(Layer):
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=1)

# Build the transformer model with masking support
def build_transformer_model(input_shape, output_dim, head_size, num_heads, ff_dim, num_transformer_blocks, dropout):
    inputs = Input(shape=input_shape)
    masking = Masking(mask_value=0.0)(inputs)

    x = Dense(64, activation='relu')(masking)

    for _ in range(num_transformer_blocks):
        x = transformer_block(x, head_size, num_heads, ff_dim, dropout)

    x = GlobalAveragePooling()(x)

    outputs = Dense(output_dim)(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


# Define the model
input_shape = (time_step, len(target_columns))
output_dim = len(target_columns)
model = build_transformer_model(input_shape, output_dim, head_size=64, num_heads=2, ff_dim=128, num_transformer_blocks=2, dropout=0.1)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

log_ram_usage("After building model")


# Record the start time before training
start_time = time.time()

# Define the checkpoint callback
checkpoint_path = "model_checkpoint.keras"
checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', mode='min')

# Train the model with checkpointing
history = model.fit(X_train, y_train, validation_split=0.2, batch_size=32, epochs=50, verbose=1, callbacks=[checkpoint])

# Record the end time after training
end_time = time.time()

# Calculate the training time
training_time = end_time - start_time

log_ram_usage("After training model")
print(f"Training time: {training_time} seconds")

# Print the model summary to see the number of parameters
model.summary()

# Alternatively, you can print the number of parameters directly
total_params = model.count_params()
print(f"Total number of parameters: {total_params}")

# Load the best model
best_model = load_model(checkpoint_path, custom_objects={'CustomMultiHeadAttention': CustomMultiHeadAttention, 'GlobalAveragePooling': GlobalAveragePooling})

# Make predictions
train_predict = best_model.predict(X_train)
test_predict = best_model.predict(X_test)

log_ram_usage("After making predictions")

# Inverse transform the predictions to get the original scale
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
original_y_train = scaler.inverse_transform(y_train.reshape(-1, len(target_columns)))
original_y_test = scaler.inverse_transform(y_test.reshape(-1, len(target_columns)))

# Calculate the root mean squared error (RMSE) for each column
train_rmse = np.sqrt(np.mean((train_predict - original_y_train) ** 2, axis=0))
test_rmse = np.sqrt(np.mean((test_predict - original_y_test) ** 2, axis=0))

print("Train RMSE: ", train_rmse)
print("Test RMSE: ", test_rmse)

log_ram_usage("After calculating RMSE")

# Function to make predictions on new data
def predict_new_data(new_data, model, scaler, time_step):
    new_data=new_data.tail()
    # Ensure new_data is a DataFrame
    if not isinstance(new_data, pd.DataFrame):
        new_data = pd.DataFrame(new_data)

    # Extract the numeric part from the ID_NXT column if necessary
    if 'ID_NXT' in new_data.columns:
        new_data['ID_NXT'] = new_data['ID_NXT'].apply(lambda x: int(re.search(r'\d+', x).group()))

    # Normalize the new data
    new_data[target_columns] = scaler.transform(new_data[target_columns])

    # Group the new data by patient ID and prepare sequences
    new_grouped = new_data.groupby('ID_NXT')
    new_sequences = []
    for name, group in new_grouped:
        new_sequences.append(group[target_columns].values)

    # Pad sequences to the same length as the training data
    new_padded_sequences = pad_sequences(new_sequences, padding='post', dtype='float32', maxlen=time_step)

    # Prepare the data for the transformer model
    new_X = []
    for sequence in new_padded_sequences:
        if len(sequence) >= time_step:
            new_X.append(sequence[-time_step:])
        else:
            # If the sequence is shorter than the time_step, pad it with zeros
            new_X.append(np.pad(sequence, ((time_step - len(sequence), 0), (0, 0)), mode='constant', constant_values=0))

    new_X = np.array(new_X)
    print(new_X)
    # Make predictions
    predictions = model.predict(new_X)

    # Inverse transform the predictions to get the original scale
    predictions = scaler.inverse_transform(predictions)

    return predictions

# Example usage with new data
new_data_path = '/content/sample_new_patient_data.csv'
new_data = pd.read_csv(new_data_path)

log_ram_usage("After loading new data")

# Get predictions
predictions = predict_new_data(new_data, model, scaler, time_step)

# Display predictions
if predictions is not None:
    print(predictions)
else:
    print("No predictions were made due to invalid input sequences.")

log_ram_usage("After making predictions for new data")


# Function to plot existing and new predicted data
def plot_predictions(new_data, predictions, target_columns):
    fig, axes = plt.subplots(len(target_columns), 1, figsize=(10, 5 * len(target_columns)),  dpi=300)

    if len(target_columns) == 1:
        axes = [axes]

    for i, column in enumerate(target_columns):
        years = new_data['APP_YEAR'].unique()
        #years.append(new_data['APP_YEAR'][len(years)]+1)
        new_value = years[-1] + 1

        years = np.append(years, new_value)
        preds = np.append(new_data[column],predictions[0][i])
        axes[i].plot(years, preds, label='Predictions', marker='x')
        axes[i].set_title(f'{column}')
        axes[i].set_xlabel('Year')
        axes[i].set_ylabel('Value')
        axes[i].legend()

    plt.tight_layout()
    plt.show()

# Plot the new predictions
plot_predictions(pd.read_csv(new_data_path), predictions, target_columns)

# Function to plot existing and new predicted data
def validate_data(new_data, predictions, target_columns):
    new_data=new_data.head()
    fig, axes = plt.subplots(len(target_columns), 1, figsize=(10, 5 * len(target_columns)),  dpi=300)
    known = new_data.head(4)
    if len(target_columns) == 1:
        axes = [axes]

    for i, column in enumerate(target_columns):
        years = new_data['APP_YEAR'].unique()
        new_value = years[-1] + 1
        all_years = np.append(years, new_value)

        # Append the predictions to the actual values
        predicted_values = np.append(known[column].values, predictions[0][i])
        all_values = new_data[column].values

        if len(all_values) == len(predicted_values):
            rmse = np.sqrt(mean_squared_error(all_values, predicted_values))
            print(f'RMSE for {column}: {rmse:.2f}')
        else:
            rmse = None
            print(f'RMSE for {column}: Cannot compute RMSE due to mismatched lengths.')

        # Plot the actual values
        axes[i].plot(years, all_values, label='Actual', marker='o')

        # Plot the predictions
        axes[i].plot(years, predicted_values, label='Predictions', marker='x', linestyle='-')

        axes[i].set_title(f'{column}')
        axes[i].set_xlabel('Year')
        axes[i].set_ylabel('Value')
        axes[i].legend()

        if rmse is not None:
            axes[i].text(0.05, 0.95, f'RMSE: {rmse:.2f}', transform=axes[i].transAxes, fontsize=12,
                         verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))

    plt.tight_layout()
    plt.show()

print(predictions)
# Plot the new predictions along with the actual values
new_data_path = 'XXXXXXXXXX'
validate_data(pd.read_csv(new_data_path), predictions, target_columns)

# Define the model
input_shape = (time_step, len(target_columns))
output_dim = len(target_columns)
model = build_transformer_model(input_shape, output_dim, head_size=64, num_heads=2, ff_dim=128, num_transformer_blocks=2, dropout=0.1)

# Plot the model architecture and save it to a file
plot_model(model, to_file='transformer_model_for_paper.png', show_shapes=True, show_layer_names=True, expand_nested=True, dpi=300)

print("Model architecture diagram saved as 'transformer_model_for_paper.png'")

"""## Loading in model"""

import pickle
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer, Dense, Dropout
import pandas as pd
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

class CustomMultiHeadAttention(Layer):
    def __init__(self, head_size, num_heads, dropout=0.0, **kwargs):
        super(CustomMultiHeadAttention, self).__init__(**kwargs)
        self.head_size = head_size
        self.num_heads = num_heads
        self.dropout = dropout

    def build(self, input_shape):
        self.query_dense = Dense(self.head_size * self.num_heads)
        self.key_dense = Dense(self.head_size * self.num_heads)
        self.value_dense = Dense(self.head_size * self.num_heads)
        self.output_dense = Dense(input_shape[-1])
        self.dropout_layer = Dropout(self.dropout)

    def call(self, inputs, mask=None):
        query, key, value = inputs, inputs, inputs
        batch_size = tf.shape(query)[0]

        # Linear projections
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # Reshape to [batch_size, num_heads, seq_len, head_size]
        query = self._reshape_to_batches(query, batch_size)
        key = self._reshape_to_batches(key, batch_size)
        value = self._reshape_to_batches(value, batch_size)

        # Scaled dot-product attention
        attention_scores = tf.matmul(query, key, transpose_b=True)
        attention_scores = attention_scores / tf.sqrt(tf.cast(self.head_size, dtype=tf.float32))

        if mask is not None:
            mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype=tf.float32)
            attention_scores += (mask * -1e9)

        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_weights = self.dropout_layer(attention_weights)

        attention_output = tf.matmul(attention_weights, value)
        attention_output = self._reshape_from_batches(attention_output, batch_size)

        # Linear projection
        output = self.output_dense(attention_output)
        return output

    def compute_mask(self, inputs, mask=None):
        # Propagate the mask to downstream layers
        return mask

    def _reshape_to_batches(self, x, batch_size):
        x = tf.reshape(x, [batch_size, -1, self.num_heads, self.head_size])
        return tf.transpose(x, [0, 2, 1, 3])

    def _reshape_from_batches(self, x, batch_size):
        x = tf.transpose(x, [0, 2, 1, 3])
        return tf.reshape(x, [batch_size, -1, self.num_heads * self.head_size])

# Custom layer for global average pooling
class GlobalAveragePooling(Layer):
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=1)

# Load the scaler
scaler_path = 'scaler.pkl'
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Load the model with custom objects
checkpoint_path = "model_checkpoint.keras"
model = load_model(checkpoint_path, custom_objects={'CustomMultiHeadAttention': CustomMultiHeadAttention, 'GlobalAveragePooling': GlobalAveragePooling})

time_step=5
target_columns = []

# Function to make predictions on new data
def predict_new_data(new_data, model, scaler, time_step):
    new_data=new_data.tail()
    # Ensure new_data is a DataFrame
    if not isinstance(new_data, pd.DataFrame):
        new_data = pd.DataFrame(new_data)

    # Extract the numeric part from the ID_NXT column if necessary
    if 'ID_NXT' in new_data.columns:
        new_data['ID_NXT'] = new_data['ID_NXT'].apply(lambda x: int(re.search(r'\d+', x).group()))

    # Normalize the new data
    new_data[target_columns] = scaler.transform(new_data[target_columns])

    # Group the new data by patient ID and prepare sequences
    new_grouped = new_data.groupby('ID_NXT')
    new_sequences = []
    for name, group in new_grouped:
        new_sequences.append(group[target_columns].values)

    # Pad sequences to the same length as the training data
    new_padded_sequences = pad_sequences(new_sequences, padding='post', dtype='float32', maxlen=time_step)

    # Prepare the data for the transformer model
    new_X = []
    for sequence in new_padded_sequences:
        if len(sequence) >= time_step:
            new_X.append(sequence[-time_step:])
        else:
            # If the sequence is shorter than the time_step, pad it with zeros
            new_X.append(np.pad(sequence, ((time_step - len(sequence), 0), (0, 0)), mode='constant', constant_values=0))

    new_X = np.array(new_X)
    print(new_X)
    # Make predictions
    predictions = model.predict(new_X)

    # Inverse transform the predictions to get the original scale
    predictions = scaler.inverse_transform(predictions)

    return predictions

# Example usage with new data
new_data_path = ''
new_data = pd.read_csv(new_data_path)

# Get predictions
predictions = predict_new_data(new_data, model, scaler, time_step)

# Display predictions
if predictions is not None:
    print(predictions)
else:
    print("No predictions were made due to invalid input sequences.")


# Function to plot existing and new predicted data
def plot_predictions(new_data, predictions, target_columns):
    fig, axes = plt.subplots(len(target_columns), 1, figsize=(10, 5 * len(target_columns)),  dpi=300)

    if len(target_columns) == 1:
        axes = [axes]

    for i, column in enumerate(target_columns):
        years = new_data['APP_YEAR'].unique()
        #years.append(new_data['APP_YEAR'][len(years)]+1)
        new_value = years[-1] + 1

        years = np.append(years, new_value)
        preds = np.append(new_data[column],predictions[0][i])
        axes[i].plot(years, preds, label='Predictions', marker='x')
        axes[i].set_title(f'{column}')
        axes[i].set_xlabel('Year')
        axes[i].set_ylabel('Value')
        axes[i].legend()

    plt.tight_layout()
    plt.show()

# Plot the new predictions
plot_predictions(pd.read_csv(new_data_path), predictions, target_columns)
