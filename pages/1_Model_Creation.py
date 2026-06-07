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
classes = app_vars.classes
model = model_desc.model
X_tensor, y_tensor = dataset.tensors


col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    epochs = st.text_input('Epochs:', value='100')
    exe_container = st.container()
with col2:
    batch_size = st.text_input('Batch Size:', value='32')
with col3:
    lr = st.text_input('Learn Rate:', value='0.001')
with col4:
    test_size = st.text_input('Test percentage:', value='0.2')
    n_splits = st.text_input('Number of Splits:', value='5')
with col5:
    pos_weight = st.text_input('Positive Weight:', value='1')
    


go = exe_container.button('Create a new model!')

if not go:
    st.stop()


st.write(f"Overall Prediction Accuracy of {class_name} on {app_vars.study} using {model_desc.X_desc} as features:")
summery_empty = st.empty()
summery_empty.progress(0.01)
summary_container = st.container()


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

    
    if class_name == MODEL_SINGLE:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(int(pos_weight)))
        optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))

        torch_train_batch(model, criterion, optimizer, int(epochs), train_subset.dataset, int(batch_size))

        model.eval()
        with torch.no_grad():
            # y_train_pred = model(X_train_tensor)
            y_test_pred = model(X_test_tensor)
          
        


        c1, c2 = st.columns(2)
        with c1:
            probs = torch.sigmoid(y_test_pred).detach().numpy()
            preds = (probs > 0.5).astype(int)
        
            roc_auc = round(roc_auc_score(y_test_tensor, probs), 3)
            pr_auc = round(average_precision_score(y_test_tensor, probs),3)
            accuracy = round(accuracy_score(y_test_tensor, preds), 3)

            st.write(f'roc_auc = {roc_auc} | pr_auc = {pr_auc} | accuracy = {accuracy}')

            report = get_classification_report(y_test_tensor.numpy(), preds)

            del report['accuracy']
            st.dataframe(report.transpose())


        with c2:
            fig, ax = plt.subplots(figsize=(5, 2.5),  layout="constrained")
            ConfusionMatrixDisplay.from_predictions(y_test_tensor.numpy(), preds, labels=None, display_labels=None, ax=ax, colorbar=True)
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            st.image(buf)
            # st.write('Confusion Matrix. X - Prediction; Y - Actual')

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
        
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_dataloader = DataLoader(test_dataset, shuffle=False)
        
        all_preds, all_targets = evaluate_multitask_model(model, test_dataloader, device, classes)

        # display_multitask_results(all_preds, all_targets, classes)
        st_multitask_results(all_preds, all_targets, classes)
    
    summery_empty.progress((split_idx+1)/int(n_splits))
    st.write('***')
   
end = timer()
