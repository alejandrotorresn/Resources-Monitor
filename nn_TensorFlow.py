import numpy as np
import time
import pandas as pd
import tensorflow as tf


train_data = np.random.randn(5000, 32, 32, 3).astype(np.float32)
label_data = np.random.randint(0, 10, 5000).astype(np.int32)

tfk = tf.keras
tfkl = tfk.layers

batch_size = 128
dataset = tf.data.Dataset.from_tensor_slices((train_data, label_data))
dataset = dataset.shuffle(buffer_size=50000)
dataset = dataset.batch(batch_size, drop_remainder=True)

times_layers = pd.DataFrame(dtype=object)
times_epochs = pd.DataFrame(dtype=object)

class Model(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.activation = tf.keras.layers.ReLU()
        self.conv1 = tfkl.Conv2D(64, (3, 3), padding='VALID')
        self.bn1 = tfkl.BatchNormalization()
        self.conv2 = tfkl.Conv2D(128, (3, 3), padding='VALID')
        self.bn2 = tfkl.BatchNormalization()
        self.conv3 = tfkl.Conv2D(256, (3, 3), padding='VALID')
        self.bn3 = tfkl.BatchNormalization()
        self.flatten = tfkl.Flatten()
        self.dense = tfkl.Dense(10)
        
    def call(self, inputs, training=False):
        global times_layers
        time_layer = pd.DataFrame(dtype=object)
        time_l = []

        time_l.append(time.time())
        h = self.conv1(inputs)
        time_l.append(time.time())
        h = self.bn1(h, training=training)
        time_l.append(time.time())
        h = self.activation(h)
        time_l.append(time.time())
        h = self.conv2(h)
        time_l.append(time.time())
        h = self.bn2(h, training=training)
        time_l.append(time.time())
        h = self.activation(h)
        time_l.append(time.time())
        h = self.conv3(h)
        time_l.append(time.time())
        h = self.bn3(h, training=training)
        time_l.append(time.time())
        h = self.activation(h)
        time_l.append(time.time())
        h = self.flatten(h)

        time_layer = pd.DataFrame(time_l)
        print(time_l)
        times_layers = times_layers.append(time_layer.T, ignore_index=True)

        return self.dense(h)

train_loss = tfk.metrics.Mean() 
train_acc = tfk.metrics.SparseCategoricalAccuracy()
optimizer = tf.optimizers.Adam(1.0e-4)
model = Model()

@tf.function
def train_step(inputs):
    images, labels = inputs
    
    with tf.GradientTape() as tape:
        logits = model(images, training=True)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
        
    grad = tape.gradient(loss, sources=model.trainable_variables)
    optimizer.apply_gradients(zip(grad, model.trainable_variables))
    
    train_loss.update_state(loss)
    train_acc.update_state(labels, logits)


epochs = 10

for epoch in range(epochs):

    time_start = time.time()
    times_epochs = times_epochs.append([time_start])
    for images, labels in dataset:
        with tf.device('/device:GPU:0'):
            train_step((images, labels)) 

    epoch_loss = train_loss.result()
    epoch_acc = 100 * train_acc.result()

    time_epoch = time.time() - time_start
    print('epoch: {:} loss: {:.4f} acc: {:.2f}% time: {:.2f}s'.format(epoch + 1, epoch_loss, epoch_acc, time_epoch))

    train_loss.reset_states()
    train_acc.reset_states()

times_layers.to_csv('reports/times_layers_report_tf.csv', index=False)
times_epochs.to_csv('reports/times_epochs_report_tf.csv', index=False)

