import tensorflow as tf
from typing import Literal

PI = 3.141592653589793

def nll(eval_pts,sim,bw):
    '''
    Computes negative log likelihood

    Parameters
    ----------

    eval_pts : array-like
        Points to evaluate the KDE at
        Must be of dimension (N,D)
        N is # of data points, D # of channels
    
    sim : array-like
        Simulation used to create the KDE
        Must be of dimension (M,D)
        M is # of data points, D # of channels
    
    bw : float
        Bandwidth
    '''

    # Convert all to same dtype
    eval_pts = tf.convert_to_tensor(eval_pts)
    sim = tf.cast(sim,eval_pts.dtype)
    bw = tf.cast(bw,eval_pts.dtype)

    n = tf.cast(tf.shape(sim)[0], eval_pts.dtype) # Num data pts
    d = tf.cast(tf.shape(sim)[1], eval_pts.dtype) # Num channels

    diff = eval_pts[:, None, :] - sim[None, :, :] # Pairwise difference
    dist_sq = tf.reduce_sum(tf.square(diff),axis=-1) # Compute square

    log_kernels = -dist_sq/(2.0 * bw ** 2)
    log_sum = tf.reduce_logsumexp(log_kernels,axis=1)

    log_norm = tf.math.log(n) + 0.5 * d * tf.math.log(tf.constant(2.0*PI,dtype=eval_pts.dtype)) + d * tf.math.log(bw)

    log_kde = log_sum - log_norm

    return -tf.reduce_sum(log_kde)