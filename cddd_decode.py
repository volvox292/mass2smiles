import pandas as pd
from cddd.inference import InferenceModel
from cddd.preprocessing import preprocess_smiles
import numpy as np
import sys
import os

inference_model = InferenceModel()
fname6=os.path.join(sys.argv[1],"result_predict.npy")
#x=np.load('/Users/delser/mass2smiles/output_loss_predicted.npy')
x=np.load(fname6)

#flat=x[:, :, 0]
decoded_smiles_list = inference_model.emb_to_seq(x)

if type(decoded_smiles_list)== str:
    decoded_smiles_list=[decoded_smiles_list]
else:
    pass
df_cddd_decoded = pd.DataFrame(decoded_smiles_list)
#df_cddd_decoded=df_cddd_decoded.drop(["Unnamed: 0"], axis=1)
#fname7=os.path.join(sys.argv[1],"output_loss_predicted.tsv")
#df.to_csv("/Users/delser/mass2smiles/output_loss_predicted.tsv",sep='\t')
#df.to_csv(fname7,sep='\t')


fname8=os.path.join(sys.argv[1],'feature_ids_dataframe.tsv')
df_samples = pd.read_csv(fname8, sep="\t") # mgf dataframe with loss

#df_samples = pd.read_csv("/Users/delser/mass2smiles/output_loss.tsv", header=None, sep="\t") # mgf dataframe with loss

#fname9=os.path.join(sys.argv[1],"output_loss_predicted.tsv")

#df_pred = pd.read_csv(fname9,index_col= "Unnamed: 0",sep="\t") # predicted smiles from cddd
#df_pred = pd.read_csv("/Users/delser/mass2smiles/output_loss_predicted.tsv",index_col= "Unnamed: 0",sep="\t") # predicted smiles from cddd

df_cddd_decoded=df_cddd_decoded.rename(columns={"0": "predicted"})
#df_final = df_samples.join(df_pred, how="outer")
#df_final=df_final.drop(["Column 3"], axis=1)
#df_final=df_final.drop([1,2,3], axis=1)

df_final = pd.concat([df_samples, df_cddd_decoded ], axis=1)
df_final=df_final.drop(["Unnamed: 0"], axis=1)
df_final=df_final.drop(["mzs","intensities",'loss_mzs','loss_intensities'], axis=1)
#df_final=df_final.rename(columns={0: "feature_ID"})
fname9=os.path.join(sys.argv[1],"result_predict1.npy")
x1=np.load(fname9)

def float_oupt_to_class(oupt, k):
  end_pts = np.zeros(k+1, dtype=np.float32) 
  delta = 1.0 / k
  for i in range(k):
    end_pts[i] = i * delta
  end_pts[k] = 1.0
  # if k=4, [0.0, 0.25, 0.50, 0.75, 1.0] 

  for i in range(k):
    if oupt >= end_pts[i] and oupt <= end_pts[i+1]:
      return i
  return -1  # fatal error 
  
funct=['#num_of_sugars',"#Number of aliphatic carboxylic acids",
 "#Number of aliphatic hydroxyl groups",
 "#Number of aliphatic hydroxyl groups excluding tert-OH",
 "#Number of N functional groups attached to aromatics",
 "#Number of Aromatic carboxylic acides",
 "#Number of aromatic nitrogens",
 "#Number of aromatic amines",
 "#Number of aromatic hydroxyl groups",
 "#Number of carboxylic acids",
 "#Number of carboxylic acids",
 "#Number of carbonyl O",
 "#Number of carbonyl O, excluding COOH",
 "#Number of thiocarbonyl",
 "#Number of Imines",
 "#Number of Tertiary amines",
 "#Number of Secondary amines",
 "#Number of Primary amines",
 "#Number of hydroxylamine groups",
 "#Number of tert-alicyclic amines (no heteroatoms, not quinine-like bridged N)",
 "#Number of H-pyrrole nitrogens",
 "#Number of thiol groups",
 "#Number of aldehydes",
 "#Number of alkyl carbamates (subject to hydrolysis)",
 "#Number of alkyl halides",
 "#Number of allylic oxidation sites excluding steroid dienone",
 "#Number of amides",
 "#Number of anilines",
 "#Number of aryl methyl sites for hydroxylation",
 "#Number of azo groups",
 "#Number of benzene rings",
 "#Bicyclic",
 "#Number of dihydropyridines",
 "#Number of epoxide rings",
 "#Number of esters",
 "#Number of ether oxygens (including phenoxy)",
 "#Number of furan rings",
 "#Number of guanidine groups",
 "#Number of halogens",
 "#Number of imidazole rings",
 "#Number of isothiocyanates",
 "#Number of ketones",
 "#Number of ketones excluding diaryl, a,b-unsat. dienones, heteroatom on Calpha",
 "#Number of beta lactams",
 "#Number of cyclic esters (lactones)",
 "#Number of methoxy groups -OCH3",
 "#Number of nitriles",
 "#Number of nitro groups",
 "#Number of oxazole rings",
 "#Number of para-hydroxylation sites",
 "#Number of phenols",
 "#Number of phosphoric acid groups",
 "#Number of phosphoric ester groups",
 "#Number of piperdine rings",
 "#Number of primary amides",
 "#Number of pyridine rings",
 "#Number of quaternary nitrogens",
 "#Number of thioether",
 "#Number of thiazole rings",
 "#Number of unbranched alkanes of at least 4 members (excludes halogenated alkanes)",
      "#adduct_enc","#C","#H", "#O","#N", "#S", "#I", "#Br","#Cl", "#F","#P"]

all_arr=[]
for sample in x1:
    cache=[]
    for i in sample:
        cache.append(float_oupt_to_class(i,65))
    xn=np.array(cache)
    all_arr.append(xn)
    
result=np.stack(all_arr)   
#result.shape  

df = pd.DataFrame(result, columns = funct)

df_final = df_final.join(df, how="outer")




fname10=os.path.join(sys.argv[1],"predicted_results.tsv")

df_final.to_csv(fname10,sep='\t')

#df_final.to_csv("/Users/delser/mass2smiles/predicted_results.tsv",sep='\t')