from PIL import Image
import PIL.Image
from IPython.display import Image, display
import cv2
from scipy import ndimage

from matplotlib.pyplot import imread, imshow

import sys 
import os
from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem import Descriptors as d
from rdkit.Chem import Lipinski as l
from rdkit.Chem import MolSurf as surf
from rdkit.Chem import Atom as at
from rdkit.Chem import AllChem
from rdkit.Chem import ChemicalFeatures as cf
from rdkit.Chem import DataStructs as cdat
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import rdDepictor as rd
from rdkit.Chem import Draw

# This file contains all the main external libs we'll use
from fastai import *
from fastai.vision import *
import torchvision.models as tvm

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144

import time
from datetime import datetime, timedelta


import numpy as np
import pandas as pd

import glob
import pickle

import os

# directory stuff
import glob

from mordred import Calculator, descriptors

from sklearn.externals import joblib


## 1) IMPORT FUNCTIONS USED
# def register_extension(id, extension): Image.EXTENSION[extension.lower()] = id.upper()
# Image.register_extension = register_extension
# def register_extensions(id, extensions): 
#     for extension in extensions: register_extension(id, extension)
# Image.register_extensions = register_extensions

def mordred_descriptors(molsmile, descriptors_length):
    """
    Args:
        List: a list with Mol from SMILES data
        int: the length of the number of descriptors from Mordred = 1613
        
    Returns:
        Pandas Dataframe: a dataframe of the descriptors only from the mols, index reset
    """
    calc = Calculator(descriptors, ignore_3D=True)
    
    
    nan_values = np.nan * np.empty((1, descriptors_length))
    df = pd.DataFrame()
    
    if molsmile == None:
        df = df.append(pd.DataFrame(nan_values, columns=df.columns))
    else:
        df = df.append(calc.pandas([molsmile]))  

    df = df.reset_index()
    df = df.drop(['index'], axis=1)
    
    mordred_descriptors = pd.DataFrame(df)

    return mordred_descriptors

def fillNA_withMedian(dataframe):
    BBBP_descriptors = dataframe.apply(pd.to_numeric, errors='coerce')
    column_medians = BBBP_descriptors.median(axis=0)
    imputed_descriptors = BBBP_descriptors.fillna(column_medians)
    imputed_descriptors = imputed_descriptors.fillna(0)
    return imputed_descriptors

## 2) COLLECT THE SMILES DATA
SMILES = input('SMILES: ')
molsmile = Chem.MolFromSmiles(SMILES)
if molsmile == None:
    print('SMILES could not be converted to mol form')


## 3) CONVERT SMILES TO AN IMAGE
Draw.MolToFile(molsmile, f'test_images/%s.png' %SMILES, size = (224, 224))

print('predicting')

# 4) FOR USE IN THE RFC MODEL - PROBLEM - IMPUTED MEDIAN VALUES
mordred = mordred_descriptors(molsmile, 1613)
to_drop = pickle.load(open(f'dropped_no_topo.p', 'rb'))
mordred = mordred.drop(to_drop, axis=1)
imputed_descriptors = fillNA_withMedian(mordred) # get the median from the original dataset I was working on 

loaded_model = joblib.load(f'RFC_5CV_noTopo_06cutoff.sav')
result = loaded_model.predict(imputed_descriptors)
proba  = loaded_model.predict_proba(imputed_descriptors)
print('RF Classifier Prediction:', result, proba[:,1])

## 5) FOR USE IN THE CNN PREDICTION
small_drug_ids_labels = pd.read_csv(f'split_set_1/png_small_drug_ids_labels.csv')



data = ImageDataBunch.from_df(f'split_set_1/small_images', small_drug_ids_labels,
                            valid_pct= 0.2, size=224)

learn = create_cnn(data, models.resnet34)
learn.load('split_set_1/fastai_v1_resnet34')

fn = f'test_images/%s.png' %SMILES
image = open_image(fn)

pred_class, pred_idx, probs = learn.predict(image)
prob = probs[0].numpy()

print('CNN Classifier Prediction:', [pred_class], prob)

