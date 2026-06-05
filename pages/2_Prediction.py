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
    st.write(f'No {model_desc.class_name} model with {model_desc.X_desc} features are available for {app_vars.study}')
    st.stop()
   
# load the trained model
model_key = f'{prefix}model_desc.pkl'
model_desc:ModelDesc = get_from_s3(env.s3_bucket, model_key)

col1, col2 = st.columns([1,2])

smiles = ''
df_input = None
with col1:
    smiles_list = []
    cmpd_list = []
    exp_val_list = []

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

            exp_col_0 = st.selectbox('Select Experiment val Column if available:', options=col_all)  
            if exp_col_0 != '--':
                if  logarithmic_scale:
                    exp_col = 'log_' + exp_col_0
                    df_input[exp_col] = df_input[exp_col_0].apply(lambda x: math.log10(x))
                else:
                    exp_col = exp_col_0

                exp_val_list = df_input[exp_col].tolist() 


    df_pred = None
    if smiles_list:
        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        
        X =  get_all_descriptors(mols, radius=RADIUS, fp_size=FP_SIZE, descriptor_sel=model_desc.X_desc, reduced=False)
        X = X[model_desc.X_cols]
        X = model_desc.X_scaler.transform(X)

        if model_desc.class_name == MODEL_TORCH:
            model = model_desc.model
            model.eval()
            with torch.no_grad():
                y_pred = model(torch.tensor(X, dtype=torch.float32)).numpy()
        else:
            y_pred = model_desc.model.predict(X)
            y_pred = y_pred.reshape(-1,1)

        y_pred = model_desc.y_scaler.inverse_transform(y_pred)
        preds = y_pred.reshape(-1)
    else:
        # w/o SMILES list, nothing can be done
        st.stop()  
        
    
with col2:

    if exp_val_list:
        expt_label = exp_col
        pred_label = f'pred_{expt_label}'
        if cmpd_list:
            list_of_tuples = list(zip(cmpd_list, smiles_list, exp_val_list, preds))
            df_pred = pd.DataFrame(list_of_tuples, columns=['Compound_ID', 'SMILES', expt_label, pred_label])
        else:
            list_of_tuples = list(zip(smiles_list, exp_val_list, preds))
            df_pred = pd.DataFrame(list_of_tuples, columns=['SMILES', expt_label, pred_label])
    else:
        expt_label=''
        pred_label = f'pred_{app_vars.expt_col_name}'
        if cmpd_list:
            list_of_tuples = list(zip(cmpd_list, smiles_list, preds))
            df_pred = pd.DataFrame(list_of_tuples, columns=['Compound_ID', 'SMILES', pred_label])
        else:
            list_of_tuples = list(zip(smiles_list, preds))
            df_pred = pd.DataFrame(list_of_tuples, columns=['SMILES', pred_label])
            

    row_id = df_pred.index.to_numpy()
    df_pred.insert(loc=0, column='row_id', value=row_id)

    if expt_label:
        highlight_only = st.checkbox('Only display selected mol in the correlation fig')    
    df_container = st.container()

   
    
    if expt_label and expt_label in df_pred.columns:
        st.write(f"The average K-fold (20/80) R2 of the model = {model_desc.k_fold_r2}")
        st.write(f'The R2 for this data file is {round(r2_score(df_pred[expt_label], df_pred[pred_label]),2)}')
        fig_df_structure(df_pred, expt_label, pred_label, df_container, mol_container, highlight_only=highlight_only)
    else:
        st.dataframe(df_pred, hide_index=True)
