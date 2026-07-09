import streamlit as st
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from sklearn.metrics import classification_report,  ConfusionMatrixDisplay
from torch.utils.data import TensorDataset, Subset, DataLoader
import matplotlib.pyplot as plt
from ptc_util import *
from ptc_comp import *
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from timeit import default_timer as timer
from datetime import timedelta
import io
from sklearn import preprocessing


# start = timer()

if 'new_model' in st.session_state:
    new_model = st.session_state['new_model']
else:
    st.write('Please go back to home page to set up a model to create.')
    st.stop()


if not new_model:
    st.write(f"An existing model will be used.")
    st.stop()

env:Env = None
if 'env' in st.session_state:
    env = st.session_state['env']

app_vars:AppVars = None
if 'app_vars' in st.session_state:
    app_vars = st.session_state['app_vars']

model_desc:ModelDesc = None
if 'model_desc' in st.session_state:
    model_desc = st.session_state['model_desc']

dataset:TensorDataset = None
if 'dataset' in st.session_state:
    dataset = st.session_state['dataset']


model_class = model_desc.model_class
classes = app_vars.classes
# model = model_desc.model
# model.to(DEVICE)
X_tensor, y_tensor = dataset.tensors

save_to_master = False
# copy_container = None

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    epochs = st.text_input('Epochs:', value='100')
    exe_container = st.container()
    exe_empty = st.empty()
with col2:
    batch_size = st.text_input('Batch Size:', value='32')
with col3:
    lr = st.text_input('Learn Rate:', value='0.001')
with col4:
    test_size = st.text_input('Test percentage:', value='0.2')
    n_splits = st.text_input('Number of Splits:', value='5')
with col5:
    pos_weight = st.text_input('Positive Weight:', value='1')

    if app_vars.is_admin:
        save_to_master = st.checkbox('Save to master folder!')
    #     copy_container = st.container()
    


go = exe_container.button('Create a new model!')

if not go:
    st.stop()

exe_empty.write("Training/Evaluating train/test sets ...")

summery_empty = st.empty()
summery_empty.progress(0.01)

sss = None
if model_class in [MODEL_L3, MODEL_L4]:
    sss= StratifiedShuffleSplit(
        n_splits=int(n_splits),
        test_size=float(test_size),
        random_state=42
    )
elif model_class == MODEL_MULTI:
    sss = MultilabelStratifiedShuffleSplit(
        n_splits=int(n_splits),
        test_size=float(test_size),
        random_state=42
    )

X_np = X_tensor.numpy()
y_np = y_tensor.numpy()
y_np_filled = np.nan_to_num(y_np, nan=0.0)  

for split_idx, (train_idx, test_idx) in enumerate(sss.split(X_np, y_np_filled)):

    X_train, X_test = X_np[train_idx], X_np[test_idx]
    y_train, y_test = y_np[train_idx], y_np[test_idx]

    X_scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)
    
    if model_class in [MODEL_L3, MODEL_L4]:
        model = model_factory(model_class, input_dim=len(model_desc.X_cols))
        model.to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight)))
        optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))

        torch_train_batch(model, 
                          criterion, 
                          optimizer, 
                          int(epochs), 
                          TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)), 
                          int(batch_size))

        # model.eval()
          
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
        test_dataloader = DataLoader(test_dataset, shuffle=False)
        
        all_preds, all_targets = evaluate_model(model, test_dataloader)


        c1, c2 = st.columns(2)
        with c1:
            st_result_matrix(all_targets, all_preds)

        with c2:
            st_confusion_matrix(all_targets, all_preds)

    elif model_class == MODEL_MULTI:
        model = model_factory(model_class, input_dim=len(model_desc.X_cols))
        model.to(DEVICE)
        criterion = MaskedBCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight)))
        optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
       
        # for name, param in model.named_parameters():
        #     if torch.isnan(param).any():
        #         print(f"{name} contains NaN")

        torch_train_batch(model, 
                          criterion, 
                          optimizer, 
                          int(epochs), 
                          TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)), 
                          int(batch_size))

        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
        test_dataloader = DataLoader(test_dataset, shuffle=False)
        
        all_preds, all_targets = evaluate_model(model, test_dataloader)

        st_multitask_results(all_preds, all_targets, classes)
    
    summery_empty.progress((split_idx+1)/int(n_splits))
    st.write('***')

exe_empty.write("Training with whole dataset ...")
# Train with the whole dataset and save it to S3

X_scaler = preprocessing.StandardScaler().fit(X_np)
X_np = X_scaler.transform(X_np)

if model_class in [MODEL_L3, MODEL_L4]:
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight)))

elif model_class == MODEL_MULTI:
    criterion = MaskedBCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight)))

# No need to re-initialize model; current model from last fold training might be helpful

optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
torch_train_batch(model, 
                  criterion, 
                  optimizer, 
                  int(epochs), 
                  TensorDataset(torch.tensor(X_np, dtype=torch.float32), torch.tensor(y_np, dtype=torch.float32)), 
                  int(batch_size))


model.to('cpu')
model_desc.X_scaler = X_scaler
model_desc.model = model   # populate model_desc with trained model

# key_prefix = f'{app_vars.login_name}/{env.app_data}/{app_vars.study}/{model_class}/{model_desc.X_desc}'
if save_to_master:
    key_prefix = get_prefix_master(env, app_vars, model_desc)
else:
    key_prefix = get_prefix(env, app_vars, model_desc)


key_model_desc = f'{key_prefix}model_desc.pkl'
pickle_to_s3(model_desc, env.s3_bucket, key_model_desc)
key_app_vars = f'{key_prefix}app_vars.pkl'
pickle_to_s3(app_vars, env.s3_bucket, key_app_vars)

exe_empty.write(f"Done! And model saved.")



# end = timer()
