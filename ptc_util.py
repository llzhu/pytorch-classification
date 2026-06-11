import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score, classification_report
from rdkit import Chem, DataStructs
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor,Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.Descriptors import rdFingerprintGenerator
from io import StringIO, BytesIO
import os
import shutil
import re
from dataclasses import dataclass, field
from typing import List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader 
from icecream import ic
import base64
from itertools import chain
import boto3
import pickle
from typing import Tuple


s3client = boto3.client(
    "s3",
    aws_access_key_id = st.secrets['aws_access_key_id'],
    aws_secret_access_key = st.secrets['aws_secret_access_key'],
    region_name = st.secrets['region_name']
)

s3resource = boto3.resource(
    "s3",
    aws_access_key_id = st.secrets['aws_access_key_id'],
    aws_secret_access_key = st.secrets['aws_secret_access_key'],
    region_name = st.secrets['region_name']
)

def pickle_to_s3(data_obj, bucket, key):
    # Serialize to memory
    buffer = BytesIO()
    pickle.dump(data_obj, buffer)
    buffer.seek(0)

    # Upload to S3
    s3client.put_object(
        Bucket=bucket,
        Key=key,
        Body=buffer.getvalue()
    )

def get_from_s3(bucket, key):
    # Download pickle from S3
    pickle_obj = s3client.get_object(Bucket=bucket, Key=key)

    # Deserialize
    buffer = BytesIO(pickle_obj["Body"].read())
    data_obj = pickle.load(buffer)

    return data_obj

def any_contents(bucket, prefix):
    response = s3client.list_objects_v2(
        Bucket=bucket,
        Prefix=prefix,
        Delimiter="/"
    )

    # ic(response.get("Contents", []))
    return len(response.get("Contents", []))>0
    
    

def get_df_from_s3csv(bucket, key):
    # Get object from S3
    obj = s3client.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(BytesIO(obj["Body"].read()))
    return df


# algorithms
MODEL_MULTI = 'Multi-Task'
MODEL_SINGLE = 'Single Task'

MODEL_OPTIONS = [ MODEL_MULTI, MODEL_SINGLE]

# discriptors
FP_ONLY = 'Morgan_FP'
ADD_RDKIT_DESCRIPTORS = 'Morgan_FP_2D_Descriptors'
RDKIT_DESCRIPTORS_ONLY = '2D_Descriptors'

FEATURE_OPTIONS = [ADD_RDKIT_DESCRIPTORS, FP_ONLY, RDKIT_DESCRIPTORS_ONLY]

# Studies/dataset
TOX21 = 'Tox21'
TOX21_NR_AR = 'Tox21_NR-AR'
TOX21_NR_AR_LBD = 'Tox21_NR-AR-LBD'
TOX21_NR_AHR = 'Tox21_NR-AhR'
TOX21_NR_AROMATASE = 'Tox21_NR-Aromatase'
TOX21_NR_ER = 'Tox21_NR-ER'
TOX21_NR_ER_LBD = 'Tox21_NR-ER-LBD'
TOX21_NR_PPAR_GAMMA = 'Tox21_NR-PPAR-gamma'
TOX21_SR_ARE = 'Tox21_SR-ARE'
TOX21_SR_ATAD5 = 'Tox21_SR-ATAD5'
TOX21_SR_HSE = 'Tox21_SR-HSE'
TOX21_SR_MMP = 'Tox21_SR-MMP'
TOX21_SR_P53 = 'Tox21_SR-p53'

AD_HOC = 'ad_hoc'
TOX21_ALL_CLASSES = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase','NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
                    'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
TOX21_DICT = {'NR-AR':0, 
              'NR-AR-LBD':1, 
              'NR-AhR':2, 
              'NR-Aromatase':3,
              'NR-ER':4, 
              'NR-ER-LBD':5, 
              'NR-PPAR-gamma':6,
              'SR-ARE':7, 
              'SR-ATAD5':8, 
              'SR-HSE':9, 
              'SR-MMP':10, 
              'SR-p53':11}

STUDY_OPTIONS = ['--', TOX21, TOX21_NR_AR, TOX21_NR_AR_LBD, TOX21_NR_AHR, TOX21_NR_AROMATASE, TOX21_NR_ER, TOX21_NR_ER_LBD,
                    TOX21_NR_PPAR_GAMMA, TOX21_SR_ARE, TOX21_SR_ATAD5, TOX21_SR_HSE, TOX21_SR_MMP, TOX21_SR_P53, AD_HOC]

RADIUS = 3 
FP_SIZE = 4096
LEARNING_RATE = 0.001

SMI_LIST = 'SMILES lists'
FILE_UPLOAD = 'File Upload'

SMILES = 'SMILES'
DO_NOT_HIGHLIGHT = "Do not highlight"
HIGHLIGHT_ALL = "Highlight All"
HIGHLIGHT_UNIQUE = "Highlight Unique"
COMPOUND_ID = 'Compound_ID'
STRUCTURE = 'Compound'
CHEMBL_UNIT = 'standard_units'
CHEMBL_SMILES = 'canonical_smiles'
CHEMBL_CMPD_ID = 'molecule_chembl_id'


@dataclass
class AppVars:
    login_name: str
    is_admin: bool
    study: str
    dataset_shape: Tuple[int,int] = (1, 1)
    classes: List[str] = field(default_factory=list)
    
    
@dataclass
class ModelDesc:
    X_desc: str = ''
    X_cols: List[str] = field(default_factory=list)
    X_scaler: StandardScaler = None
    model_class: str = ''
    model: object = None
  

@dataclass
class ModelData:
    X: pd.DataFrame
    y: List[float]

@dataclass
class Env:
    src_data: str
    app_data: str
    admins: List[str] = field(default_factory=list)
    modelers: List[str] = field(default_factory=list)
    s3_bucket: str = ''



def get_prefix(env:Env, app_vars:AppVars, model_desc:ModelDesc):
    return f'{app_vars.login_name}/{env.app_data}/{app_vars.study}/{model_desc.model_class}/{model_desc.X_desc}/'

def get_prefix_master(env:Env, app_vars:AppVars, model_desc:ModelDesc):
    return f'{env.app_data}/{app_vars.study}/{model_desc.model_class}/{model_desc.X_desc}/'

def delete_contents(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def get_df_csv(df):
    f = StringIO()
    df.to_csv(f, index=False)
    return f


def convert_df_csv(df, index=False):
    return df.to_csv(index=index).encode('utf-8')



def copy_s3_folder(bucket_name, source_folder, destination_folder):
    """
    Copies all files from one folder to another within the same S3 bucket.
    """
    
    # Ensure folder paths end with a trailing slash
    if not source_folder.endswith('/'):
        source_folder += '/'
    if not destination_folder.endswith('/'):
        destination_folder += '/'
        
    # Use a paginator to handle folders containing more than 1,000 objects
    paginator = s3client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=source_folder)
    
    for page in page_iterator:
        # Check if the folder contains any files
        if 'Contents' in page:
            for obj in page['Contents']:
                source_key = obj['Key']
                
                # Skip the directory placeholder itself if it exists
                if source_key == source_folder:
                    continue
                
                # Construct the new destination key
                # This replaces the old prefix with the new prefix
                relative_path = source_key[len(source_folder):]
                destination_key = destination_folder + relative_path
                
                # Define the source object configuration dict
                copy_source = {
                    'Bucket': bucket_name,
                    'Key': source_key
                }
                
                print(f"Copying: {source_key} -> {destination_key}")
                
                # Perform the copy operation
                s3client.copy_object(
                    Bucket=bucket_name,
                    CopySource=copy_source,
                    Key=destination_key
                )




class L3Model(nn.Module):
    """ A DNN model with 3 layers (input, 1 hidden layer and out layer)
    """
    def __init__(self, input_dim, dim1=256, dim2=128, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, dim1),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(dim1, dim2),
            nn.ReLU(),

            nn.Linear(dim2, output_dim)
        )

    def forward(self, x):
        return self.net(x)
    
class L4Model(nn.Module):
    """ A DNN model with 4 layers (input, 2 hidden layers and out layer)
    """
    def __init__(self, input_dim, dim1=256, dim2=128, dim3=64, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, dim1),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(dim1, dim2),
            nn.ReLU(),

            nn.Linear(dim2, dim3),
            nn.ReLU(),

            nn.Linear(dim3, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def torch_train(model, epochs, X, y):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    for epoch in range(epochs):
        model.train()
    
        preds = model(X)
        loss = criterion(preds, y)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class MaskedBCEWithLogitsLoss(nn.Module):
    """  A masked BCE with logits and pos_weight 
    """
    def __init__(self, pos_weight):
        super().__init__()
        self.pos_weight = pos_weight
        
    def forward(self, predictions, targets):
        # Create a mask where targets are valid (e.g., assuming missing data is marked as -1)
        # If your data loader keeps them as NaN, use torch.isnan(targets) instead
        # mask = (targets != -1).float()
        mask = (~torch.isnan(targets)).float()

        targets = torch.nan_to_num(targets, nan=0.0)   # !! important !! nan target will result in nan raw_loss

        # ic(mask)
        # ic(targets)

        # Calculate standard Binary Cross Entropy with Logits for all elements
        # reduction='none' ensures we get a loss per individual data point
        raw_loss = F.binary_cross_entropy_with_logits(predictions, targets.float(), pos_weight=self.pos_weight, reduction='none')
        # ic(raw_loss)
        # Zero out the losses for missing labels
        masked_loss = raw_loss * mask

        # ic(masked_loss)
        
        # Return the mean loss, dividing only by the number of valid data points
        num_valid_points = torch.sum(mask)
        if num_valid_points == 0:
            return torch.tensor(0.0, requires_grad=True).to(predictions.device)
            
        return torch.sum(masked_loss) / num_valid_points
    

# A pure Python function approach; it can be used only 'stateless' pure function.
def masked_bce_loss_fn(predictions, targets):
    #mask = (targets != -1).float()
    mask = (~torch.isnan(targets)).float()
    # ic(mask)
    raw_loss = F.binary_cross_entropy_with_logits(predictions, targets.float(), reduction='none')
    masked_loss = raw_loss * mask
    
    num_valid_points = torch.sum(mask)
    if num_valid_points == 0:
        return torch.tensor(0.0, requires_grad=True, device=predictions.device)
        
    return torch.sum(masked_loss) / num_valid_points

class MultiTaskNet(nn.Module):
    def __init__(self, input_dim=1024, num_tasks=12, dim1=512, dim2=256):
        super().__init__()
        
        # 1. Shared Base Network (Learns general chemical features)
        self.shared_base = nn.Sequential(
            nn.Linear(input_dim, dim1),
            nn.ReLU(),
            nn.BatchNorm1d(dim1),
            nn.Dropout(0.2),
            
            nn.Linear(dim1, dim2),
            nn.ReLU(),
            nn.BatchNorm1d(dim2),
        )
        
        # 2. Task-Specific Heads 
        # Using nn.ModuleList allows PyTorch to track these individual layers
        self.heads = nn.ModuleList([
            nn.Linear(dim2, 1) for _ in range(num_tasks)
        ])
        
    def forward(self, x):
        # ic(x)
        # ic(x.shape)
        # Pass input through the shared chemical feature extractor
        shared_features = self.shared_base(x)
        # ic(shared_features)
        # ic(shared_features.shape)
        
        # Pass the shared features through each task head independently
        # We output raw logits (pre-sigmoid) for numerical stability during loss calculation
        outputs = [head(shared_features) for head in self.heads]
        # ic(outputs)
        # Stack outputs into a tensor of shape (batch_size, num_tasks)
        stacked_outputs =  torch.cat(outputs, dim=1)
        # ic(stacked_outputs)
        return stacked_outputs


    
# criterion = nn.MSELoss() for regression
# criterion = nn.BCELoss() or nn.BCEWithLogitsLoss() for 0/1 classification
# criteria - nn.CrossEntropyLoss() for multi-classifications

def torch_train_batch(model, criterion, optimizer, epochs, dataset, batch_size):
   
    data_loader = DataLoader(dataset, batch_size, shuffle=True)
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in data_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            # ic(preds)
            # ic(X_batch)
            loss = criterion(preds, y_batch)

            # Check for NaN before backward pass to isolate the issue
            # if torch.isnan(loss):
            #     print("Loss is NaN! Stopping training.")
            #     break
            # else:
            #     print("IT IS NOT A NAN")

            loss.backward()

            # --- GRADIENT CLIPPING FIX ---
            # Clips gradients in-place before the optimizer updates the weights
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}: {running_loss/len(dataset)}')





def evaluate_model(model, dataloader, device, num_tasks):

    model.eval()  # Set model to evaluation mode (disables dropout/batchnorm updates)
    
    all_preds = []
    all_targets = []
    
    # 1. Gather all predictions and targets across the entire dataloader
    with torch.no_grad():  # Turn off gradient tracking to save memory
        for batch_features, batch_targets in dataloader:
            batch_features = batch_features.to(device)
            
            # Forward pass to get raw logits
            logits = model(batch_features)
            
            # Convert raw logits to probabilities using Sigmoid
            probs = torch.sigmoid(logits)
            # ic(probs)
            
            all_preds.append(probs.cpu().numpy())
            all_targets.append(batch_targets.numpy())

    # ic(all_preds)
            
    # Concatenate lists into large NumPy arrays of shape (total_samples, num_tasks)
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    return all_preds, all_targets

def display_multitask_results(all_preds, all_targets, classes):

    num_tasks = len(classes)
    
    # 2. Iterate through each task to calculate individual performance metrics
    task_auc_scores = {}
    
    for task_idx in range(num_tasks):
        # Extract predictions and targets for the current task
        y_true_tensor = torch.tensor(all_targets[:, task_idx])
        y_pred_tensor = torch.tensor(all_preds[:, task_idx])
        
        # CRITICAL STEP: Create a mask to filter out missing labels (-1)
        # valid_mask = (y_true != -1)
        valid_mask = (~torch.isnan(y_true_tensor))
        # ic(valid_mask)

        y_true_filtered = y_true_tensor[valid_mask].numpy()
        y_pred_filtered = y_pred_tensor[valid_mask].numpy()
        
        # Check if the filtered task has both active (1) and inactive (0) classes
        # ROC-AUC cannot be calculated if only one class is present in the test slice
        if len(np.unique(y_true_filtered)) < 2:
            print(f"\n[Task {task_idx}] Skipped: Only one class present in valid test samples.")
            continue
            
        # Compute ROC-AUC score using predicted probabilities
        auc = roc_auc_score(y_true_filtered, y_pred_filtered)
        task_auc_scores[f"Task_{task_idx}"] = auc
        
        # For the classification report, we need binary decisions (0 or 1)
        # We use a standard threshold of 0.5
        y_pred_binary = (y_pred_filtered >= 0.5).astype(int)
        
        # Print metrics
        print(f"\n👉 TASK {classes[task_idx]} | ROC-AUC: {auc:.4f}")
        print("-" * 50)
        print(classification_report(y_true_filtered, y_pred_binary, target_names=["Inactive (0)", "Active (1)"]))
        
    # 3. Compute and print global metrics
    mean_auc = np.mean(list(task_auc_scores.values()))
    print("=" * 60)
    print(f"Overall Mean ROC-AUC across all valid tasks: {mean_auc:.4f}")
    print("=" * 60)
    
    return task_auc_scores

def get_tox21_df(classes, csv_name, env, container, model_class):
    container.write(f'The DNN is overwritten to {model_class}')
    df_g =  get_df_from_s3csv(env.s3_bucket, f'{env.src_data}/{csv_name}')
    df_g = df_g[['Title', 'SMILES'] + classes]      
    df_g = df_g.rename(columns={'Title': COMPOUND_ID,})
    return df_g

@st.cache_data
def standarize(df_input: pd.DataFrame, study:str, value_column:str, apply_log:bool)-> tuple[pd.DataFrame, str]:
    pass
    # df_output = None
    # if study == THROBIN_IC50:
    #     df_output = df_input[df_input[CHEMBL_UNIT]=='nM']
    #     df_output = df_output[~df_output[CHEMBL_SMILES].str.contains('.', regex=False, na=False)]
    #     df_output[value_column] = df_output[value_column].astype(float)
    #     if apply_log:
    #         expt_col_name = 'log_IC50'
    #         df_output[expt_col_name] = df_output[value_column].apply(math.log10)
    #     else:
    #         expt_col_name = 'IC50'
    #         df_output[expt_col_name] = df_output[value_column]

    #     column_map = {
    #             CHEMBL_CMPD_ID: COMPOUND_ID,
    #             CHEMBL_SMILES: SMILES
    #         }
    #     df_output = df_output.rename(columns=column_map) 
    #     col_output = [COMPOUND_ID, expt_col_name, SMILES]
    #     df_output = df_output[col_output] 
    #     df_output[expt_col_name] = df_output[expt_col_name].astype(float)


    # return df_output, expt_col_name




def get_list(inputs: str)->list[str]: 
    input_list = []
    if inputs:
        input_list = re.split(',|\n', inputs)
        input_list = [input for input in input_list if input.strip()]
    return input_list

def get_floor(in_num: float, floor: float)-> float: 
    out_num = in_num
    if in_num < floor:
        out_num = floor
    return out_num


def remove_low_variance(input_data, threshold=0.1) -> pd.DataFrame:
    # input_data expacted to be np.ndarray or pd.Dataframe
    if isinstance(input_data, np.ndarray):
        input_data = pd.DataFrame(data=input_data)  
    selection = VarianceThreshold(threshold)
    selection.fit(input_data)
    return input_data[input_data.columns[selection.get_support(indices=True)]]

def get_rdkit_fp(morgan_gen, mol_list):
    X = [ np.array(morgan_gen.GetFingerprint(mol)) for mol in mol_list]
    X = pd.DataFrame(data=X)  # Make it a dataframe
    return X

def get_rdkit_descriptors(mol_list, excluded_descriptors = None):
    descriptor_names = [x[0] for x in Descriptors._descList]
    if excluded_descriptors:
        descriptor_names = [d for d in descriptor_names if d not in excluded_descriptors]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    mol_descriptors = []
    index = 0
    for mol in mol_list:
        # ic(index)
        descriptors = calc.CalcDescriptors(mol)
        if np.isinf(descriptors).any():
            ic(index)
        mol_descriptors.append(descriptors)
        index = index +1

    df = pd.DataFrame(mol_descriptors, columns=descriptor_names)
    return df


def get_rdkit_descriptors_scaled(mol_list, scaler=None):
    descriptor_names = [x[0] for x in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    mol_descriptors = []
    for mol in mol_list:
        descriptors = calc.CalcDescriptors(mol)
        mol_descriptors.append(descriptors)

    if scaler == None:
        scaler = StandardScaler().fit(mol_descriptors)
        mol_descriptors = scaler.transform(mol_descriptors)
    else:
        mol_descriptors = scaler.transform(mol_descriptors)

    return pd.DataFrame(mol_descriptors, columns=descriptor_names), scaler
    

@st.cache_data
def get_all_descriptors(mol_list, radius, fp_size, descriptor_sel, reduced=True, excluded_descriptors = None):

    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)

    if descriptor_sel == FP_ONLY:
        X_FP = get_rdkit_fp(morgan_gen, mol_list)
        if reduced:
            X_FP = remove_low_variance(X_FP)
        X = X_FP


    if descriptor_sel == ADD_RDKIT_DESCRIPTORS:
        X_FP = get_rdkit_fp(morgan_gen, mol_list)
        if reduced:
            X_FP = remove_low_variance(X_FP)

        X_DESC_2D = get_rdkit_descriptors(mol_list, excluded_descriptors=excluded_descriptors)
        # ic(np.isinf(X_DESC_2D).any())
        # ic(np.isinf(X_DESC_2D.to_numpy()).sum())
        # ic(X_DESC_2D.isnull().sum())
        # ic(X_DESC_2D.describe())

        # for col in X_DESC_2D.columns:
        #     if np.isinf(X_DESC_2D[col].to_numpy()).any():
        #         print(f"Column '{col}' contains infinity values!")

        # X_DESC_2D = X_DESC_2D.replace([np.inf, -np.inf], np.nan)

        if reduced:
            X_DESC_2D = remove_low_variance(X_DESC_2D)

        X = pd.concat([X_FP, X_DESC_2D], axis=1, join='inner')


    if descriptor_sel == RDKIT_DESCRIPTORS_ONLY:
        X_DESC_2D = get_rdkit_descriptors(mol_list, excluded_descriptors=excluded_descriptors)
        if reduced:
            X_DESC_2D = remove_low_variance(X_DESC_2D)

        X = X_DESC_2D

    X.columns = X.columns.astype(str) 
    # st.dataframe(X)
    return X

def get_classification_report(y_test, y_pred):
    from sklearn import metrics
    report = metrics.classification_report(y_test, y_pred, target_names=["Inactive (0)", "Active (1)"], output_dict=True )
    df_classification_report = pd.DataFrame(report)
    return df_classification_report

@st.cache_data
def moltosvg(mol, molSize = (800,400), kekulize = False, highlight_sub=None, highlight_mode=DO_NOT_HIGHLIGHT):
    
    if  highlight_sub == None: # Cannot highlight if highlight_sub not provided
        highlight_mode=DO_NOT_HIGHLIGHT
    
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
    
    if highlight_mode == DO_NOT_HIGHLIGHT:
        drawer.DrawMolecule(mc)
    elif highlight_mode in (HIGHLIGHT_UNIQUE, HIGHLIGHT_ALL):
        highlight_tt = mc.GetSubstructMatches(highlight_sub)
        hightlight_shape = np.shape(highlight_tt)
        if hightlight_shape[0] == 1:
            highlight_tuple = tuple(chain.from_iterable(highlight_tt))
            drawer.DrawMolecule(mc, highlightAtoms=highlight_tuple)
        else:
            if highlight_mode == HIGHLIGHT_UNIQUE:
                drawer.DrawMolecule(mc)
            elif highlight_mode == HIGHLIGHT_ALL:
                highlight_tuple = tuple(chain.from_iterable(highlight_tt))
                drawer.DrawMolecule(mc, highlightAtoms=highlight_tuple)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    svg = svg.replace('svg:','')
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = rf'<img src="data:image/svg+xml;base64, {b64}"/>'
    return html