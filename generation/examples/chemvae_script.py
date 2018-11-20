# tensorflow backend
import numpy as np
import pandas as pd
import sys
print("1")
from os import environ
environ['KERAS_BACKEND'] = 'tensorflow'
# vae stuff
print("2")
from chemvae.vae_utils import VAEUtils
print("3")
from chemvae import mol_utils as mu
print("4")
# rdkit stuff
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import PandasTools
print("5")
import matplotlib as mpl
print("6")
from rdkit import Chem
from rdkit.DataStructs import cDataStructs as cdat
from rdkit import DataStructs
from rdkit.DataStructs import cDataStructs as cdat
from rdkit.Chem.Fingerprints import FingerprintMols
print("7")
# In[ ]:

vae = VAEUtils(directory='./models/zinc_properties')

def autoencode(smiles_1):
    #smiles_1 = mu.canon_smiles(smile)

    # one hot encode the smile
    X_1 = vae.smiles_to_hot(smiles_1,canonize_smiles=True)
    z_1 = vae.encode(X_1) # the latent representation of the molecule
    noise=5.0
    print("calculating......")
    df = vae.z_to_smiles(z_1,decode_attempts=1000,noise_norm=noise)
    return df['smiles'].values.tolist()

def getFingerprintSimilarity(smile1,smile2, metric_name=cdat.TanimotoSimilarity):
    """
    Args: 
        list: a list with Mol from SMILES data
        Function Call: the function call for the metric from rdkit.DataStructs import cDataStructs
        
    Return: 
        Pandas DataFrame: a particular type of fingerprint similarity matrix
    """  
    smile_list = [smile1,smile2]
    molsmiles = [Chem.MolFromSmiles(i) for i in smile_list]
    metric_names = [cdat.TanimotoSimilarity, cdat.DiceSimilarity, cdat.CosineSimilarity, cdat.SokalSimilarity]
    
    fps = [FingerprintMols.FingerprintMol(x) if x != None else 0 for x in molsmiles]

    x = np.zeros((len(molsmiles),len(molsmiles)), dtype=np.float32)
    
    fingerprints = []
    for i in range(len(x)):
        for j in range(len(x)):
            if fps[j] == 0 or fps[i] == 0:
                fingerprints.append(np.nan)
            else:
                fingerprints.append((DataStructs.FingerprintSimilarity(fps[i],fps[j], metric_name)))
                
    fingerprints = np.array(fingerprints)
    fingerprints = fingerprints.reshape(np.sqrt(len(fingerprints)).astype(int), 
                                        np.sqrt(len(fingerprints)).astype(int))
    
    fps = pd.DataFrame(fingerprints)
    
    return fps.values[0,1]


# In[ ]:

inputMol = sys.argv[1]
print("get input mol:", inputMol)
try:
    for generatedDrug in autoencode(inputMol):
        print("Generated drug structure: ", generatedDrug," Similarity: ",getFingerprintSimilarity(inputMol, generatedDrug))
except Exception:
    print("the input mor might not be valid, please check your input")
    pass

