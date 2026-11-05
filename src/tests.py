import numpy as np
import pandas as pd
from rescaler import MLSimRescaler

np.random.seed(42)

N_SIM = 10000
N_DATA = 5000

CHAN_LIST = ('PAS1','PBS1','PCS1','PDS1','PES1','PFS1',
             'PAS2','PBS2','PCS2','PDS2','PES2','PFS2')

channels = [c + "OFamps" for c in CHAN_LIST]

# Generate simulation channel distributions
sim_raw = np.random.dirichlet(
    alpha=np.ones(12) * 2,
    size=N_SIM
)

# Generate data with a slightly different distribution
data_raw = np.random.dirichlet(
    alpha=np.array([3,2,2,1.5,2,1.5,
                    2,3,2,1.5,2,1.5]),
    size=N_DATA
)

# Generate total energies
sim_psum = np.random.uniform(100, 500, N_SIM)
data_psum = np.random.uniform(100, 500, N_DATA)

# Convert normalized distributions into amplitudes
sim_amplitudes = sim_raw * sim_psum[:, None]
data_amplitudes = data_raw * data_psum[:, None]

# Construct DataFrames
sim_df = pd.DataFrame(sim_amplitudes, columns=channels)
data_df = pd.DataFrame(data_amplitudes, columns=channels)

sim_df['PSum'] = sim_psum
data_df['PSum'] = data_psum

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

inputs = layers.Input(shape=(12,))

x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)

outputs = layers.Dense(
    12,
    kernel_initializer='zeros',
    bias_initializer='zeros'
)(x)

model = models.Model(inputs, outputs)

rescaler = MLSimRescaler(
    model=model,
    sim_df=sim_df,
    data_df=data_df,
    bandwidth=0.003,
    samples_data=1000,
    samples_sim=1000
)

rescaler.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2)
)

EPOCHS = 100

for epoch in range(EPOCHS):
    rescaler.loss_tracker.reset_state()

    logs = rescaler.train_step(None)

    print(
        f"Epoch {epoch+1}/{EPOCHS} "
        f"- Loss: {logs['loss'].numpy():.6f}"
    )

result_df = rescaler.apply(sim_df)

print(result_df.head())

# Check energy conservation
original = sim_df['PSum'].to_numpy()
rescaled = result_df[channels].sum(axis=1).to_numpy()

print("Maximum energy discrepancy:",
      np.max(np.abs(original - rescaled)))

print("Mean energy discrepancy:",
      np.mean(np.abs(original - rescaled)))


import matplotlib.pyplot as plt
import numpy as np

original = sim_df[channels].to_numpy()
rescaled = result_df[channels].to_numpy()
data = data_df[channels].to_numpy()

original /= original.sum(axis=1, keepdims=True)
rescaled /= rescaled.sum(axis=1, keepdims=True)
data /= data.sum(axis=1,keepdims=True)

for i, channel in enumerate(CHAN_LIST):
    plt.figure(figsize=(7, 4))

    plt.hist(original[:, i], bins=50, alpha=0.5,
             density=True, label='Original')

    plt.hist(data[:,i], bins=50,alpha=0.5, density=True,label="Data")

    plt.hist(rescaled[:, i], bins=50, alpha=0.5,
             density=True, label='Rescaled')

    plt.title(channel)
    plt.legend()
    plt.show()