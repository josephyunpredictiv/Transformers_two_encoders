# -*- coding: utf-8 -*-
"""
Original file is located at (Not avalible publicly due to containing classified info)
    https://colab.research.google.com/drive/1D5HcDgICwmPm_Pf221be_KUs1vFgDBz3
"""

"""# Transformers"""

# Python 3.10.12

import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Masking, LayerNormalization, Dropout, MultiHeadAttention, Layer
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import psutil
import os
import time

# Function to get current RAM usage
def get_ram_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)  # Convert bytes to MB

def log_ram_usage(label):
    print(f"{label} - RAM usage: {get_ram_usage()} MB")

file_path = 'XXXXXXXXXXXXXX'
data = pd.read_csv(file_path)

log_ram_usage("After loading data")

# Extract the numeric part from the ID_NXT column
data['ID_NXT'] = data['ID_NXT'].apply(lambda x: int(re.search(r'\d+', x).group()))

# List of columns to predict
target_columns = []  # Add as many as needed

# Sort by ID_NXT and APP_YEAR
data = data.sort_values(by=['ID_NXT', 'APP_YEAR'])


# Optional Linear Interpolation. (Make sure you can assume linearity before doing.
"""
# Create a DataFrame with the full range of years for each ID_NXT
full_years_list = []

for nxt in data['ID_NXT'].unique():
    years_range = list(range(data.loc[data['ID_NXT'] == nxt, 'APP_YEAR'].min(),
                             data.loc[data['ID_NXT'] == nxt, 'APP_YEAR'].max() + 1))
    full_years_list.append(pd.DataFrame({'ID_NXT': nxt, 'APP_YEAR': years_range}))

full_years = pd.concat(full_years_list, ignore_index=True)

# Merge with the original data to fill missing years with NaN
data_full = full_years.merge(data, on=['ID_NXT', 'APP_YEAR'], how='left')

# Ensure the correct data types
data_full['SEX'] = data_full['SEX'].astype('category')
data_full['DATE_BIRTH'] = pd.to_numeric(data_full['DATE_BIRTH'], errors='coerce')

# Forward-fill and backward-fill the SEX and DATE_BIRTH columns within each ID_NXT group
data_full['SEX'] = data_full.groupby('ID_NXT')['SEX'].transform(lambda group: group.ffill().bfill())
data_full['DATE_BIRTH'] = data_full.groupby('ID_NXT')['DATE_BIRTH'].transform(lambda group: group.ffill().bfill())

# Initialize an empty DataFrame to store the interpolated target columns
interpolated_columns = pd.DataFrame(index=data_full.index)

# Interpolate missing values within each group sorted by APP_YEAR for each target column individually
for col in target_columns:
    data_full[col] = data_full.groupby('ID_NXT')[col].transform(
        lambda group: group.interpolate(method='linear').ffill().bfill()
    )

# Ensure indices are aligned correctly
data_full = data_full.reset_index(drop=True)
data=data_full
"""


# Normalize and prepare data
scaler = MinMaxScaler(feature_range=(0, 1))
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

log_ram_usage("After splitting data")


# Takes into account padding of 0. Ensures proper masking.
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

# Custom transformer block using CustomMultiHeadAttention. (For masking purposes)
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

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=50, verbose=1)

# Record the end time after training
end_time = time.time()

# Calculate the training time
training_time = end_time - start_time

log_ram_usage("After training model")
print(f"Training time: {training_time} seconds")


# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

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


# Still in works... 

# Function to preprocess new data and make predictions
"""
def predict_new_data(new_data):
    # Extract the numeric part from the ID_NXT column (ensure it matches training data encoding)
    new_data['ID_NXT'] = new_data['ID_NXT'].apply(lambda x: int(re.search(r'\d+', x).group()))

    # Sort by ID_NXT and APP_YEAR
    new_data = new_data.sort_values(by=['ID_NXT', 'APP_YEAR'])

    # Interpolate missing values within each group sorted by APP_YEAR for each target column individually
    for col in target_columns:
        new_data[col] = new_data.groupby('ID_NXT')[col].transform(
            lambda group: group.interpolate(method='linear').ffill().bfill()
        )

    # Normalize the new data
    new_data[target_columns] = scaler.transform(new_data[target_columns])

    # Group new data by patient ID
    grouped_new = new_data.groupby('ID_NXT')
    new_sequences = []
    for name, group in grouped_new:
        new_sequences.append(group[target_columns].values)

    # Pad sequences to the same length
    padded_new_sequences = pad_sequences(new_sequences, padding='post', dtype='float32')

    # Prepare the new data with the same time step
    X_new, _ = prepare_transformer_data(padded_new_sequences, time_step)

    # Make predictions
    predictions = model.predict(X_new)

    # Inverse transform the predictions to get the original scale
    predictions = scaler.inverse_transform(predictions)

    return predictions
"""

# Example usage with new data
new_data_path = 'XXXXXXX'
new_data = pd.read_csv(new_data_path)

log_ram_usage("After loading new data")

# Get predictions
predictions = predict_new_data(new_data)

# Display predictions
print(predictions)

log_ram_usage("After making predictions for new data")
