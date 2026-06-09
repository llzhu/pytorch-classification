import streamlit as st
from sklearn.metrics import r2_score
from ptc_util import *
from ptc_comp import *


if 'env' in st.session_state:
    env:Env = st.session_state['env']
else:
    st.write('Please go back to home page to set up a model to load.')
    st.stop()

if 'app_vars' in st.session_state:
    app_vars:AppVars = st.session_state['app_vars']

if 'model_desc' in st.session_state:
    model_desc:ModelDesc = st.session_state['model_desc']

    
with st.sidebar:
    mol_container = st.container()

prefix = get_prefix(env, app_vars, model_desc)
any_contents = any_contents(env.s3_bucket, prefix)

if not any_contents:
    st.write(f'No {model_desc.model_class} model with {model_desc.X_desc} features are available for {app_vars.study}')
    st.stop()
   
# load the trained model
model_key = f'{prefix}model_desc.pkl'
model_desc:ModelDesc = get_from_s3(env.s3_bucket, model_key)
app_key =  f'{prefix}app_vars.pkl'
app_vars:AppVars = get_from_s3(env.s3_bucket, app_key)

model_class = model_desc.model_class
model = model_desc.model
classes = app_vars.classes


col1, col2 = st.columns([1,2])

smiles = ''
df_input = None
with col1:
    smiles_list = []
    cmpd_list = []
    y_true = []

    mol_input = st.radio('Mol input:', [SMI_LIST, FILE_UPLOAD], horizontal=True)
    

    if mol_input == SMI_LIST:
        mols_in = st.text_area('SMILES List (separate by , or newline):', key='mols_in')
        if mols_in:
            smiles_list = get_list(mols_in)
    
    else:
        logarithmic_scale = st.checkbox('Convert to Logarithm for experimental value')
        uploaded_smiles_file = st.file_uploader("Upload a SMILES CSV file. A SMILES column is required. Expt val are optional for comparison")
        if uploaded_smiles_file:
            df_input = pd.read_csv(uploaded_smiles_file)
            col_all = df_input.columns
            col_all = col_all.insert(0, '--')
            
            smile_col = st.selectbox('Select required Smile Column:', options=col_all)
            if smile_col != '--':
                smiles_list = df_input[smile_col].tolist()
                    
            id_col = st.selectbox('Select Compund ID Column if available:', options=col_all)  
            if  id_col != '--':
                cmpd_list = df_input[id_col].tolist() 

            exp_col = st.selectbox('Select Experiment val Column if available:', options=col_all)  
            if exp_col != '--':
                y_true = df_input[exp_col].tolist() 


    df_pred = None
    if smiles_list:
        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        
        X =  get_all_descriptors(mols, radius=RADIUS, fp_size=FP_SIZE, descriptor_sel=model_desc.X_desc, reduced=False)
        X = X[model_desc.X_cols]
        X = model_desc.X_scaler.transform(X)

        model.eval()
        with torch.no_grad():
            y_pred = model(torch.tensor(X, dtype=torch.float32))

        preds = torch.sigmoid(y_pred).detach().numpy()
        preds_binary = (preds > 0.5).astype(int)
        # ic(preds)
    else:
        # w/o SMILES list, nothing can be done
        st.stop()  
        
    
with col2:

    if y_true:
        expt_label = exp_col
        ic(preds_binary)
        if len(classes) > 1:
            expt_col_index = TOX21_DICT.get(exp_col)
            preds = preds[:,  expt_col_index]
            preds_binary = (preds > 0.5).astype(int)
            ic(preds_binary)

        pred_label = f'pred_{expt_label}'
        pred_probability = f'prob_{expt_label}'
        if cmpd_list:
            list_of_tuples = list(zip(cmpd_list, smiles_list, y_true, preds, preds_binary))
            df_pred = pd.DataFrame(list_of_tuples, columns=['Compound_ID', 'SMILES', expt_label,  pred_probability, pred_label])
        else:
            list_of_tuples = list(zip(smiles_list, y_true, preds, preds_binary))
            df_pred = pd.DataFrame(list_of_tuples, columns=['SMILES', expt_label, pred_probability, pred_label])
    else:
        expt_label=''
        pred_label = 'pred_class'
        pred_probability = 'prob_class'
        if cmpd_list:
            list_of_tuples = list(zip(cmpd_list, smiles_list, preds, preds_binary))
            df_pred = pd.DataFrame(list_of_tuples, columns=['Compound_ID', 'SMILES', pred_probability, pred_label])
        else:
            list_of_tuples = list(zip(smiles_list, preds, preds_binary))
            df_pred = pd.DataFrame(list_of_tuples, columns=['SMILES', pred_probability, pred_label])

    # df_pred[pred_probability] = pd.to_numeric(df_pred[pred_probability], errors='coerce') 
    # df_pred[pred_probability] = df_pred[pred_probability].round(4)
    st.dataframe(df_pred, hide_index=True)

    if y_true:

        st_result_matrix(y_true, preds)
        st_confusion_matrix(y_true, preds)


