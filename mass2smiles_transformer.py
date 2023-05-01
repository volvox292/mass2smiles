
import tensorflow as tf
import json
import sys
import os
import subprocess
import time
start_time = time.time()
import numpy as np
import pandas as pd
import pickle
from matchms import set_matchms_logger_level
import pandas as pd
set_matchms_logger_level("ERROR")
from matchms.filtering import add_losses
from matchms.filtering import add_parent_mass
from matchms.filtering import default_filters
from matchms.filtering import normalize_intensities
from matchms.filtering import select_by_intensity
from matchms.filtering import reduce_to_number_of_peaks
from matchms.filtering import require_minimum_number_of_peaks
from matchms.filtering import select_by_mz
from matchms.importing import load_from_mgf
from matchms.exporting import save_as_mgf
from matchms.importing import load_from_msp
from matchms.filtering import repair_inchi_inchikey_smiles
from matchms.filtering import derive_inchikey_from_inchi
from matchms.filtering import derive_smiles_from_inchi
from matchms.filtering import derive_inchi_from_smiles
from matchms.filtering import harmonize_undefined_inchi
from matchms.filtering import harmonize_undefined_inchikey
from matchms.filtering import harmonize_undefined_smiles
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf
from tcn import TCN
from tensorflow import keras
from keras.initializers import glorot_uniform
##################### parse mfg and convert to df ########################
print('parsing specs now')
def spectrum_processing(s):
    """This is how one would typically design a desired pre- and post-
    processing pipeline."""
    s = default_filters(s)
    s = add_parent_mass(s)
    s = normalize_intensities(s)
    s = select_by_intensity(s, intensity_from=0.01)
    s = reduce_to_number_of_peaks(s, n_required=5, n_max=250)
    s = select_by_mz(s, mz_from=15, mz_to=2000)
    s = add_losses(s, loss_mz_from=15.0, loss_mz_to=350.0)
    s = require_minimum_number_of_peaks(s, n_required=5)
    return s



def metadata_processing(spectrum):
    spectrum = default_filters(spectrum)
    spectrum = repair_inchi_inchikey_smiles(spectrum)
    spectrum = derive_inchi_from_smiles(spectrum)
    spectrum = derive_smiles_from_inchi(spectrum)
    spectrum = derive_inchikey_from_inchi(spectrum)
    spectrum = harmonize_undefined_smiles(spectrum)
    spectrum = harmonize_undefined_inchi(spectrum)
    spectrum = harmonize_undefined_inchikey(spectrum)
    return spectrum
# Load data from MGF file and apply filters


#path_data =   # enter path to downloaded mgf file
file_mgf = os.path.join(sys.argv[2], 
                         sys.argv[1])
spectrums = list(load_from_mgf(file_mgf))

spectrums = [metadata_processing(s) for s in spectrums]
spectrums = [spectrum_processing(s) for s in spectrums]
#spectrums = [spectrum_processing(s) for s in load_from_mgf("/Users/delser/Desktop/PhD/Phytochemistry/NP-Databases/CFM-4_DB/TOTAL_COMPOUNDS_DB.energies_merged_name.mgf")]
#spectrums = [spectrum_processing(s) for s in load_from_mgf("/Users/delser/Desktop/PhD/Phytochemistry/FBMN/alltissues/altissues15072021-py.mgf")]
# Omit spectrums that didn't qualify for analysis
spectrums = [s for s in spectrums if s is not None]

precs = []
IDs = []
mzs=[]
ints=[]
loss_mzs=[]
loss_ints=[]


for spec in spectrums: 
    IDs.append(spec.get("feature_id"))
    precs.append(spec.get("precursor_mz"))
    mzs.append(list(spec.peaks.mz))
    ints.append(list(spec.peaks.intensities))
    loss_mzs.append(list(spec.losses.mz))
    loss_ints.append(list(spec.losses.intensities))

metadata = pd.DataFrame(list(zip(IDs, precs,mzs,ints,loss_mzs,loss_ints)), columns=["feature_id", "precursor_mz","mzs","intensities","loss_mzs","loss_intensities" ])
fname2=os.path.join(sys.argv[2],'feature_ids_dataframe.tsv')
metadata.to_csv(fname2,sep='\t')
print("done!")
##################### encode specs ########################
print('encoding specs now')
def positional_encoding(max_position, d_model, min_freq=1e-6):
    position = np.arange(max_position)
    freqs = min_freq**(2*(np.arange(d_model)//2)/d_model)
    pos_enc = position.reshape(-1,1)*freqs.reshape(1,-1)
    pos_enc[:, ::2] = np.cos(pos_enc[:, ::2])
    pos_enc[:, 1::2] = np.sin(pos_enc[:, 1::2])
    return pos_enc

def trun_n_d(n,d):
    return (  n if not n.find('.')+1 else n[:n.find('.')+d+1]  )

P=positional_encoding(200000,256, min_freq=1e2)

def prepro_specs_train(df):
    valid=[]
    precs=df['precursor_mz'].to_list()
    mzs=df['mzs'].to_list()
    ints=df['intensities'].to_list()
    loss_mzs=df['loss_mzs'].to_list()
    loss_ints=df['loss_intensities'].to_list()
    for one_pre,one_mzs,one_ints,one_loss,one_loss_ints in zip(precs,mzs,ints,loss_mzs,loss_ints):
        mz_list=[round(float(trun_n_d(str(one_pre),2))*100)] # add precursor mz
        intes_list=[2.0] # add precursor int
        res = dict(zip(one_mzs+one_loss, one_ints+one_loss_ints))  # order by mzs
        res=dict(sorted(res.items()))
        for m,i in zip(list(res.keys()), list(res.values())): # change this from mgf from matchms
            mz=round(float(trun_n_d(str(m),2))*100)
            mz_list.append(mz)
            intens=round(i,4)
            intes_list.append(intens)
        int_mzs=[intes_list,mz_list]   
        valid.append(int_mzs) # put intesities at first
    return tf.ragged.constant(valid)

train=prepro_specs_train(metadata)

dimn=256
def encoding(rag_tensor,P,dimn):
    to_pad=[]
    for sample in rag_tensor:
        all_dim=[sample[0].numpy().tolist()]
        pos_enc=[P[int(i)-1] for i in sample[1].numpy().tolist()]
        for dim in range(dimn):
            dim_n=[i[dim] for i in pos_enc]
            all_dim.append(dim_n)
        to_pad.append(all_dim)
    to_pad=[tf.keras.preprocessing.sequence.pad_sequences(i,maxlen=501,dtype='float32',padding='post',truncating='post',value=10) for i in to_pad]
    to_pad=np.stack((to_pad))
    to_pad=np.swapaxes(to_pad, 1, -1)
    return to_pad

xtrain=encoding(train,P,dimn)
print("done!")
#xval=np.load('/home/delser/train/tcn/casmi_specs.npy')
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
  def __init__(self, x_data, y_data,y1_data, batch_size):
    self.x, self.y, self.y1 = x_data, y_data, y1_data,
    self.batch_size = batch_size
    self.num_batches = np.ceil(len(x_data) / batch_size)
    self.batch_idx = np.array_split(range(len(x_data)), self.num_batches)

  def __len__(self):
    return len(self.batch_idx)

  def __getitem__(self, idx):
    batch_x = self.x[self.batch_idx[idx]]
    batch_y = self.y[self.batch_idx[idx]]
    batch_y1 = self.y1[self.batch_idx[idx]]
    return batch_x, [batch_y,batch_y1]




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
    output_b=Sequential([               
                Dropout(rate=dense_dropout),
                Dense(128, activation='tanh'),
                Dropout(rate=dense_dropout),              
                Dense(71, activation='sigmoid')],name="funct_groups")(hd_tcn)
    output_a = Sequential([               
            Dropout(rate=dense_dropout),
            Dense(512, activation='relu'),
            Dropout(rate=dense_dropout),              
            Dense(512, activation='linear')
        ],name="smiles")(hd_tcn)
    model= tf.keras.Model(inputs=input0, outputs=[output_a,output_b])
    model.compile(loss={"smiles":'mean_absolute_error',"funct_groups":'mse'}, optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),metrics={"smiles":'mean_squared_error',"funct_groups":'mean_absolute_error'})
    
        
    return model

def build_model():
    num_layers = 5
    units = 2048
    heads = 16
    dropout = 0.1
    dense_dropout = 0.1
    filters=256
    #activation = hp.Choice("activation", ["relu", "tanh"])
    #dropout = hp.Boolean("dropout")
    #lr = hp.Float("lr", min_value=1e-6, max_value=1e-4, sampling="log")
    # call existing model-building code with the hyperparameter values.
    #model = call_existing_code(units=units, heads=heads, dropout=dropout,dense_dropout=dense_dropout,lr=lr,filters=filters)
    model = call_existing_code(units=units, heads=heads, dropout=dropout,dense_dropout=dense_dropout,lr=1e-4,filters=filters, num_layers =num_layers)
    return model



##################### predict and decode ########################
model=build_model()
model.load_weights(os.path.normpath(sys.argv[2]+"/misunderstood-fire-207/model"))
#model = keras.models.load_model(os.path.normpath(sys.argv[2]+"/upbeat-puddle-198"), custom_objects={'TCN': TCN, 'GlorotUniform': glorot_uniform()})
#model.summary()
result= model.predict(xtrain)
np.save(os.path.join(sys.argv[2],"result_predict.npy"), result[0])
np.save(os.path.join(sys.argv[2],"result_predict1.npy"), result[1])


print('predict with  transformer done!')

###### cddd decode predictions #####
print("decode embeddings now!")
x=subprocess.check_output(['conda', 'run','-n', 'cddd', 'python', 'app/cddd_decode.py',sys.argv[2]])
print(x.decode('ascii')) 

print("done!")
print("Everything was successfully predicted!")
print("Everything was successfully predicted in --- %s minutes --- to the file predicted_results.tsv" % ((time.time() - start_time)/60))
