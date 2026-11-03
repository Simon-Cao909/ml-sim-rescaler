from kde import nll
from sklearn.neighbors import KernelDensity
import numpy as np
import tensorflow as tf

simulation = tf.Variable(
    tf.random.normal((1000, 2), dtype=tf.float64)
)

data = tf.random.normal((1000, 2), dtype=tf.float64)

bw = 0.003

kde = KernelDensity(
    kernel='gaussian',
    bandwidth=bw
).fit(simulation.numpy())

logp = kde.score_samples(data.numpy())

print("Sklearn NLL:", -np.sum(logp))
print("TensorFlow NLL:", nll(data, simulation, bw).numpy())