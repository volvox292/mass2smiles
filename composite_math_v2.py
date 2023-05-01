import os
import numpy as np
from collections import defaultdict
from matchms import Spikes
import math
#import deepchem as dc
#dc.__version__
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

import pickle
path_data = "/home/delser/" 
outfile = os.path.join(path_data, 'bmdms_cache.pickle')
with open(outfile, 'rb') as file:
    spectrums = pickle.load(file)
    
def spectrum_processing(s):
    """This is how one would typically design a desired pre- and post-
    processing pipeline."""
    s = normalize_intensities(s)
    s = select_by_intensity(s, intensity_from=0.01)
    s = reduce_to_number_of_peaks(s, n_required=4, n_max=500)
    return s 

spectrums = [spectrum_processing(s) for s in spectrums]
spectrums = [s for s in spectrums if s is not None]

for i in spectrums:
    adduct=i.metadata['precursortype']
    i._metadata['adduct']=adduct
    
for i in spectrums:
    i._metadata['spectrumid']=i.metadata['comment'].split(";")[0]
    


    
def compare_update(first_mz_intensity_dict,second_mz_intensity_dict):
    modified_dict=first_mz_intensity_dict.copy()
    for key in first_mz_intensity_dict:
        for key_2 in second_mz_intensity_dict:
            if math.isclose(key,key_2,abs_tol=0.005) == False:
                mzs_modified_dict=list(modified_dict.keys())
                matches=[math.isclose(i,key_2,abs_tol=0.005) for i in mzs_modified_dict]
                if True not in matches:
                    modified_dict[key_2]=second_mz_intensity_dict[key_2]
            else:
                if first_mz_intensity_dict[key]<second_mz_intensity_dict[key_2]:
                    modified_dict[key]=second_mz_intensity_dict[key_2]
    return modified_dict


modified_spectra = defaultdict(list)

for count,i in enumerate(spectrums):
    print(count)
    if i.metadata['name'] not in modified_spectra:
        modified_spectra[i.metadata['name']]=i
    else:
        first_mz_intensity_dict=dict(zip(modified_spectra[i.metadata['name']].peaks.mz, modified_spectra[i.metadata['name']].peaks.intensities))
        second_mz_intensity_dict=dict(zip(i.peaks.mz, i.peaks.intensities))
        update=compare_update(first_mz_intensity_dict,second_mz_intensity_dict)
        update_sorted=dict(sorted(update.items()))
        modified_spectra[i.metadata['name']].peaks=Spikes(mz=np.array(list(update_sorted.keys()), dtype="float"), intensities=np.array(list(update_sorted.values()), dtype="float"))
        
mod_specs_matchms=list(modified_spectra.values())

pickle.dump(mod_specs_matchms, 
            open(os.path.join(path_data,'compositebmdms_math_500.pickle'), "wb"))
        