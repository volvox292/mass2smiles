import pandas as pd
import numpy as np
from cddd.inference import InferenceModel
from cddd.preprocessing import preprocess_smiles

ames_df = pd.read_csv("/home2020/home/ibmp/delser/cddd/nist/all_HRMS_train_24012023_cddd_refine_s.tsv", index_col="spectrum_id",sep="\t")
ames_df["smiles"] = ames_df.smiles_preprocessed.map(preprocess_smiles)
ames_df = ames_df.dropna()
smiles_list = ames_df["smiles"].tolist()

inference_model = InferenceModel()
print("Encoding now!")
smiles_embedding = inference_model.seq_to_emb(smiles_list)
print("Saving file")
np.save('/home2020/home/ibmp/delser/cddd/nist/cddd_all_HRMS_train_24012023_cddd_refine.npy', smiles_embedding)


print("Done!")


ames_df = pd.read_csv("/home2020/home/ibmp/delser/cddd/nist/all_HRMS_valid_24012023_cddd_refine_s.tsv", index_col="spectrum_id",sep="\t")
ames_df["smiles"] = ames_df.smiles_preprocessed.map(preprocess_smiles)
ames_df = ames_df.dropna()
smiles_list = ames_df["smiles"].tolist()

inference_model = InferenceModel()
print("Encoding now!")
smiles_embedding = inference_model.seq_to_emb(smiles_list)
print("Saving file")
np.save('/home2020/home/ibmp/delser/cddd/nist/cddd_all_HRMS_valid_24012023_cddd_refine.npy', smiles_embedding)

print("Done!")
