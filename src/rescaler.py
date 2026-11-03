import tensorflow as tf
from tensorflow.keras import models as km
from tensorflow.keras import metrics as kmet
import pandas as pd

from kde import nll

CHAN_LIST = ('PAS1','PBS1','PCS1','PDS1','PES1','PFS1',
             'PAS2','PBS2','PCS2','PDS2','PES2','PFS2')

class MLSimRescaler(km.Model):
    def __init__(self,
                 model,
                 sim_df,
                 bandwidth: float = 0.003,
                 samples_data: int = 1000,
                 samples_sim: int = 1000):

        super().__init__()
        self.sim_df = sim_df
        self.bandwidth = bandwidth
        self.samples_data = samples_data
        self.samples_sim = samples_sim

        self.model = model

        self.loss_tracker = kmet.Mean()

    def process(self,data_df):
        data_df = data_df.copy()
        sim_df = self.sim_df.copy()

        if 'PSum' not in sim_df.columns:
            sim_df['PSum'] = sim_df[[c+"OFamps" for c in CHAN_LIST]].sum(axis=1)

        if 'PSum' not in data_df.columns:
            data_df['PSum'] = data_df[[c+"OFamps" for c in CHAN_LIST]].sum(axis=1)

        # Shuffle rows
        sim_df = sim_df.sample(frac=1, ignore_index=True)
        data_df = data_df.sample(frac=1, ignore_index=True)

        self.sim_psum = sim_df['PSum']
        self.data_psum = data_df['PSum']
        self.norm_sim = sim_df[[c+"OFamps" for c in CHAN_LIST]].to_numpy().reshape((len(sim_df),12))/self.sim_psum
        self.norm_data = data_df[[c+"OFamps" for c in CHAN_LIST]].to_numpy().reshape((len(data_df),12))/self.data_psum

    @property
    def metrics(self):
        return [self.loss_tracker]

    def get_loss(self,data,sim):
        sim = self.model(sim)
        norm_psum = tf.reduce_sum(sim,axis=1)

        # Need to maintain constant total energy
        loss = nll(data,sim,self.bandwidth) + tf.reduce_sum(tf.square(norm_psum - 1.0))

        return loss

    def train_step(self,data_df):
        self.process(data_df)

        data = self.norm_data[:self.samples_data]
        sim = self.norm_sim[:self.samples_sim]

        data = tf.convert_to_tensor(data)
        sim = tf.convert_to_tensor(sim)
        
        with tf.GradientTape as tape:
            loss = self.get_loss(data,sim)

        gradient = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradient, self.trainable_weights))

        self.loss_tracker.update_state(loss)
        
        return {'loss':self.loss_tracker.result()}

    def test_step(self,data_df):
        self.process(data_df)

        data = self.norm_data[:self.samples_data]
        sim = self.norm_sim[:self.samples_sim]

        data = tf.convert_to_tensor(data)
        sim = tf.convert_to_tensor(sim)

        loss = self.get_loss(data,sim)
        self.loss_tracker.update_state(loss)

        return {'loss':self.loss_tracker.result()}

    def call(self,inputs):
        return self.model(inputs)

    def apply(self):
        sim_df = self.sim_df.copy()

        if 'PSum' not in sim_df.columns:
            sim_df['PSum'] = sim_df[[c+"OFamps" for c in CHAN_LIST]].sum(axis=1)

        norm_sim = sim_df[[c+"OFamps" for c in CHAN_LIST]].to_numpy().reshape((len(sim_df),12))/sim_df['PSum']

        pred = self.model(norm_sim)
        unnorm = pred * sim_df['PSum'].to_numpy()[:, None]

        for ind,c in enumerate(CHAN_LIST):
            sim_df[c+"OFamps"] = unnorm[:, ind]

        return sim_df
            