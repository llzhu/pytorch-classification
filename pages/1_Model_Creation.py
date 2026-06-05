import streamlit as st
import numpy as np
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from sklearn.metrics import classification_report,  ConfusionMatrixDisplay
from torch.utils.data import TensorDataset, Subset, DataLoader
import matplotlib.pyplot as plt
from ptc_util import *
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from timeit import default_timer as timer
from datetime import timedelta
import io


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

start = timer()

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


class_name = model_desc.class_name
model = model_desc.model
X_tensor, y_tensor = dataset.tensors


col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    epochs = st.text_input('Epochs:', value='100')
with col2:
    batch_size = st.text_input('Batch Size:', value='32')
with col3:
    lr = st.text_input('Learn Rate:', value='0.001')
with col4:
    test_size = st.text_input('Test percentage:', value='0.2')
    n_splits = st.text_input('Number of Splits:', value='5')
with col5:
    pos_weight = st.text_input('Positive Weight:', value='1')
    


go = st.button('Create a new model!')

if not go:
    st.stop()


st.write(f"Overall Prediction Accuracy of {class_name} on {app_vars.study} using {model_desc.X_desc} as features:")
summery_empty = st.empty()
summery_empty.progress(0.01)
summary_container = st.container()



st.write(f'Detailed R2 scores and Root mean square error for different train/test selections')

sss = None

if class_name == MODEL_SINGLE:
    sss= StratifiedShuffleSplit(
        n_splits=int(n_splits),
        test_size=float(test_size),
        random_state=42
    )
elif class_name == MODEL_MULTI:
    sss = MultilabelStratifiedShuffleSplit(
        n_splits=int(n_splits),
        test_size=float(test_size),
        random_state=42
    )

X_np = X_tensor.numpy()
y_np = y_tensor.numpy()
y_np_filled = np.nan_to_num(y_np, nan=0.0)  

for split_idx, (train_idx, test_idx) in enumerate(sss.split(X_np, y_np_filled)):

    # Create subsets for this specific fold
    train_subset = Subset(dataset, train_idx)
    test_subset = Subset(dataset, test_idx)      

    X_train_tensor = train_subset.dataset.tensors[0][train_subset.indices]
    y_train_tensor = train_subset.dataset.tensors[1][train_subset.indices]
    X_test_tensor = test_subset.dataset.tensors[0][test_subset.indices]
    y_test_tensor = test_subset.dataset.tensors[1][test_subset.indices]

    

    st.write(f"Split {split_idx+1} -----------------------------------")
    
    if class_name == MODEL_SINGLE:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(int(pos_weight)))
        optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))

        torch_train_batch(model, criterion, optimizer, int(epochs), train_subset.dataset, int(batch_size))

        model.eval()
        with torch.no_grad():
            y_train_pred = model(X_train_tensor)
            y_test_pred = model(X_test_tensor)
          
        probs = torch.sigmoid(y_test_pred).detach().numpy()
        # ic(probs)
        roc_auc = roc_auc_score(y_test_tensor, probs)
        pr_auc = average_precision_score(y_test_tensor, probs)

        preds = (probs > 0.5).astype(int)

        # f1 = f1_score(y_test_tensor, preds)
        accuracy = accuracy_score(y_test_tensor, preds)

        st.write(f'roc_auc = {roc_auc}')
        st.write(f'pr_auc = {pr_auc}')
        st.write(f'accuracy = {accuracy}')


        c1, c2 = st.columns(2)
        with c1:
            report = get_classification_report(y_test_tensor.numpy(), preds)

            report = report.rename(columns = {'0.0':'0', '1.0':'1'})
            del report['accuracy']
            st.dataframe(report.transpose())


        with c2:
            fig, ax = plt.subplots(figsize=(3, 2))
            ConfusionMatrixDisplay.from_predictions(y_test_tensor.numpy(), preds, labels=None, ax=ax, colorbar=True)
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            st.image(buf)
            st.write('Confusion Matrix. X - Prediction; Y - Actual')

    elif class_name == MODEL_MULTI:
        
        criterion = MaskedBCEWithLogitsLoss(pos_weight=torch.tensor(int(pos_weight)))
        # criterion = masked_bce_loss_fn
        optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
       
        # for name, param in model.named_parameters():
        #     if torch.isnan(param).any():
        #         print(f"{name} contains NaN")

        torch_train_batch(model, criterion, optimizer, int(epochs), train_subset.dataset, int(batch_size))

        model.eval()
        with torch.no_grad():
            y_train_pred = model(X_train_tensor)
            y_test_pred = model(X_test_tensor)
          
        probs = torch.sigmoid(y_test_pred).detach().numpy()
        
        test_dataloader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False)
        evaluate_multitask_model(model, test_dataloader, device, num_tasks=12)
    

    st.write('***')
   
end = timer()
