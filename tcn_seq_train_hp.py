from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten,MultiHeadAttention
from tensorflow import keras
import tensorflow as tf
import numpy as np
from tcn import TCN
#import wandb
#from wandb.keras import WandbCallback
import pickle
import os
import keras_tuner

#wandb.init(project="mass2smiles-tcn_seq")

# Run before every test for reproducibility

def seed_all():

    np.random.seed(123)

    tf.random.set_seed(123)
    
seed_all()
    

batch_size=16
#n_epochs=100


ytr=np.load('/home/delser/train/tcn/cddd_all_HRMS_train_24012023_cddd_refine.npy')
#ytr=np.expand_dims(ytr,-1)
#ytr1=np.load('/home/delser/train/tcn/y1_all_HRMS_train_24012023_cddd.npy')

yval=np.load('/home/delser/train/tcn/cddd_all_HRMS_valid_24012023_cddd_refine.npy')
#yval1=np.load('/home/delser/train/tcn/y1_all_HRMS_valid_24012023_cddd.npy')
#yval=np.expand_dims(yval,-1)

xtr=np.load('/home/delser/train/tcn/tcn_train_seq_sin256_2401.npy',mmap_mode='r')
xval =np.load( '/home/delser/train/tcn/tcn_valid_seq_sin256_2401.npy',mmap_mode='r')



class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()
        
class FeedForward(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model),
      tf.keras.layers.Dropout(dropout_rate)
    ])
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x) 
    return x
    
class GlobalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x):
    x = self.self_attention(x)
    x = self.ffn(x)
    return x

class DataGenerator(keras.utils.Sequence):
  def __init__(self, x_data, y_data, batch_size):
    self.x, self.y = x_data, y_data
    self.batch_size = batch_size
    self.num_batches = np.ceil(len(x_data) / batch_size)
    self.batch_idx = np.array_split(range(len(x_data)), self.num_batches)

  def __len__(self):
    return len(self.batch_idx)

  def __getitem__(self, idx):
    batch_x = self.x[self.batch_idx[idx]]
    batch_y = self.y[self.batch_idx[idx]]
    return batch_x, batch_y

train_generator = DataGenerator(xtr, ytr, batch_size = 16)



def call_existing_code(units, heads, dropout,dense_dropout,lr,filters,num_layers):
    tcn=TCN(
            nb_filters=filters,
            kernel_size=8,
            dilations=[2 ** i for i in range(6)],
            use_skip_connections=True,
            use_layer_norm=True,
            kernel_initializer='glorot_uniform',
            go_backwards=True,)
    print(f'TCN.receptive_field: {tcn.receptive_field}.')
    input0=tf.keras.Input(shape=(501,257))
    input1=tf.keras.layers.Masking(mask_value=10,input_shape=(501,257))(input0)
    att = Sequential([               
            EncoderLayer(d_model=257, num_heads=heads, dff=units,dropout_rate=dropout) 
        for _ in range(num_layers)])(input1)
    hd_tcn=tcn(att)
    output_a = Sequential([               
            Dropout(rate=dense_dropout),
            Dense(512, activation='relu'),
            Dropout(rate=dense_dropout),              
            Dense(512, activation='linear')
        ],name="smiles")(hd_tcn)
    model= tf.keras.Model(inputs=input0, outputs=output_a)
    model.compile(loss={"smiles":'mean_absolute_error'}, optimizer=tf.keras.optimizers.Adam(learning_rate=lr),metrics={"smiles":'mean_squared_error'})
        
    return model

def build_model(hp):
    num_layers = hp.Int("num_layers", min_value=2, max_value=6, step=2)
    units = hp.Int("units", min_value=512, max_value=2048, step=512)
    heads = hp.Int("heads", min_value=8, max_value=32, step=8)
    dropout = hp.Float("dropout", min_value=0.1, max_value=0.5, step=0.1)
    dense_dropout = hp.Float("dense_dropout", min_value=0.1, max_value=0.5, step=0.1)
    filters=hp.Int("filters", min_value=128, max_value=512, step=128)
    #activation = hp.Choice("activation", ["relu", "tanh"])
    #dropout = hp.Boolean("dropout")
    #lr = hp.Float("lr", min_value=1e-6, max_value=1e-4, sampling="log")
    # call existing model-building code with the hyperparameter values.
    #model = call_existing_code(units=units, heads=heads, dropout=dropout,dense_dropout=dense_dropout,lr=lr,filters=filters)
    model = call_existing_code(units=units, heads=heads, dropout=dropout,dense_dropout=dense_dropout,lr=1e-4,filters=filters, num_layers =num_layers)
    return model

tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective=keras_tuner.Objective("val_mean_squared_error", direction="min"),
    max_trials=99,
    executions_per_trial=1,
    overwrite=True,
    directory="my_dir",
    project_name="helloworld",
)

tuner.search_space_summary()
tuner.search(train_generator, epochs=4, validation_data=(xval,yval),batch_size=None,validation_batch_size=16)
#model.compile(loss={"smiles":'mean_absolute_error',"funct_groups":'mse'}, optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),metrics={"smiles":'mean_squared_error',"funct_groups":'mean_absolute_error'})
#model.summary()
#model.fit(train_generator,validation_data=(xval,[yval,yval1]), epochs=n_epochs,shuffle=False,batch_size=None,validation_batch_size=16,callbacks=[WandbCallback(log_batch_frequency=1)])
#result= model.predict(xval)
#np.save("/home/delser/train/tcn/val_predict.npy", result[0])
#np.save("/home/delser/train/tcn/val_predict1.npy", result[1])

#model.save('/home/delser/train/tcn')

print('done!')