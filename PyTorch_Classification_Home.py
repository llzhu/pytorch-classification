import pandas as pd
import streamlit as st
import plotly.figure_factory as ff
from rdkit import Chem
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from torch.utils.data import TensorDataset, DataLoader 
from ptc_util import *
from ptc_comp import *


st.set_page_config(page_title='PyTorch Classification', layout='wide')


env = Env(  st.secrets['src_data'],
            st.secrets['app_data'],
            st.secrets['admins'],       
            st.secrets['modelers'],
            st.secrets['s3_bucket'],
        ) 


app_header()

login_name, study, X_desc, algorithm, excluded_list, exclusion_seed, excluded_pct, new_model, algorithm_container = app_setup() 

st.session_state['new_model'] = new_model
st.session_state['env'] = env

model = None

if study == '--':
    st.error('You must select a dataset to create/upload a model.' )
    st.stop()
       
if not new_model:
    # These basic data are still need to properly load the existing models
    app_vars = AppVars(login_name=login_name, is_admin=login_name in env.admins,study=study)
    model_desc = ModelDesc(X_desc=X_desc, model_class=algorithm, model=model)
    st.session_state['app_vars'] = app_vars   
    st.session_state['model_desc'] = model_desc

    st.write(f"An existing model for {app_vars.study} will be used.")
    st.stop()

ic(study)
ic(algorithm)
if (study == TOX21 and algorithm != MODEL_MULTI) or (study != TOX21 and algorithm == MODEL_MULTI):
    st.error(f'{MODEL_MULTI} applies and only applies to {TOX21}')
    st.stop()

warning_container = st.container()

df_g = None
bin_0 = [0.1]
classes = []


if study == TOX21:
    algorithm = MODEL_MULTI
    classes = TOX21_ALL_CLASSES
    # df_g = get_tox21_df(classes, 'tox21_single_organic_nn.csv', env)
    df_g = get_tox21_df(classes, 'Tox21.csv', env)
elif study == TOX21_NR_AR:
    classes = ['NR-AR']
    df_g = get_tox21_df(classes, 'tox21_nr_ar.csv', env)
elif study == TOX21_NR_AR_LBD:
    classes = ['NR-AR-LBD']
    df_g = get_tox21_df(classes, 'tox21_nr_ar_lbd.csv', env)  
elif study == TOX21_NR_AHR:
    classes = ['NR-AhR']
    df_g = get_tox21_df(classes, 'tox21_nr_ahr.csv', env) 
elif study == TOX21_NR_AROMATASE:
    classes = ['NR-Aromatase']
    df_g = get_tox21_df(classes, 'tox21_nr_aromatase.csv', env) 
elif study == TOX21_NR_ER:
    classes = ['NR-ER']
    df_g = get_tox21_df(classes, 'tox21_nr_er.csv', env) 
elif study == TOX21_NR_ER_LBD:
    classes = ['NR-ER-LBD']
    df_g = get_tox21_df(classes, 'tox21_nr_er_lbd.csv', env)  
elif study == TOX21_NR_PPAR_GAMMA:
    classes = ['NR-PPAR-gamma']
    df_g = get_tox21_df(classes, 'tox21_nr_ppar_gamma.csv', env) 
elif study == TOX21_SR_ARE:
    classes = ['SR-ARE']
    df_g = get_tox21_df(classes, 'tox21_sr_are.csv', env) 
elif study == TOX21_SR_ATAD5:
    classes = ['SR-ATAD5']
    df_g = get_tox21_df(classes, 'tox21_sr_atad5.csv', env)
elif study == TOX21_SR_HSE:
    classes = ['SR-HSE']
    df_g = get_tox21_df(classes, 'tox21_sr_hse.csv', env)
elif study == TOX21_SR_MMP:
    classes = ['SR-MMP']
    df_g = get_tox21_df(classes, 'tox21_sr_mmp.csv', env)
elif study == TOX21_SR_P53:
    classes = ['SR-p53']
    df_g = get_tox21_df(classes, 'tox21_sr_p53.csv', env)

elif study == AD_HOC:
    df_g, expt_col_name = side_data_file_upload(warning_container=warning_container)
    
df_ex = None
if excluded_pct and int(excluded_pct) > 0:
    # ss = ShuffleSplit(n_splits=1, test_size=int(excluded_pct)/100.0, random_state=int(exclusion_seed))
    if algorithm == MODEL_L3 or algorithm == MODEL_L4:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=int(excluded_pct)/100.0, random_state=int(exclusion_seed))
    elif algorithm == MODEL_MULTI:
        sss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=int(excluded_pct)/100.0, random_state=int(exclusion_seed))
        
    for train_index, test_index in sss.split(X=df_g[SMILES].to_numpy(), y=np.nan_to_num(df_g[classes].to_numpy())):
        df_ex = df_g.iloc[test_index]
        df_g = df_g.iloc[train_index]
        
elif excluded_list:
    df_ex = df_g[df_g[COMPOUND_ID].isin(excluded_list)]
    df_g = df_g[~df_g[COMPOUND_ID].isin(excluded_list)]
    
if df_g is not None:
    csv = convert_df_csv(df_g)
    st.sidebar.download_button("Download data file", data=csv, file_name=f'data_{study}.csv', mime='text/csv')
if df_ex is not None:
    csv_ex = convert_df_csv(df_ex)
    st.sidebar.download_button("Download excluded data file", data=csv_ex, file_name=f'excluded_{study}.csv', mime='text/csv')

chem_list = [Chem.MolFromSmiles(smiles) for smiles in df_g.SMILES]
excluded_descriptors = ['Ipc']
X = get_all_descriptors(chem_list, radius=RADIUS, fp_size=FP_SIZE, descriptor_sel=X_desc, reduced=True, excluded_descriptors=excluded_descriptors)
# X is DataFrame at this point
X_cols = X.columns

# X_scaler = preprocessing.StandardScaler().fit(X)
# X = X_scaler.transform(X)   # X is ndarray at this point
X = X.values  # do not scale, get values directly

y = df_g[classes].values

# Any null values in X ????
# X_df = pd.DataFrame(data=X, columns=X_cols)

# if X_df.isinf().any().any():
#     print(f"Found {X_df.isnull().sum().sum()} NaN values!")
#     nan_positions = np.argwhere(X_df.isnull().values)
#     print(nan_positions)
# else:
#     print("=========== There is no null value in X")
#     print(X.min())
#     print(X.max())


# if algorithm == MODEL_MULTI:
#     model = MultiTaskNet(input_dim=len(X_cols), num_tasks=len(classes), dim1=128, dim2=64)

# elif algorithm == MODEL_L3:
#     model = L3Model(input_dim=len(X_cols), dim1=512, dim2=64, output_dim=1)

# elif algorithm == MODEL_L4:
#     model = L4Model(input_dim=len(X_cols), dim1=512, dim2=128, dim3=32, output_dim=1)

# model.to(device)

app_vars = AppVars(login_name=login_name, is_admin=login_name in env.admins,study=study,dataset_shape=X.shape, classes=classes)
model_desc = ModelDesc(X_desc=X_desc, X_cols=X_cols, model_class=algorithm)
dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))

# save them to session state
st.session_state['app_vars'] = app_vars   
st.session_state['model_desc'] = model_desc
st.session_state['dataset'] = dataset


c_data, c_fig = st.columns(2)
with c_data:
    st.write('All Feature Shapes = ', X.shape)
    st.dataframe(df_g)

with c_fig:
    for col in classes:
        list_col = list(df_g[col].values)
        list_col = [c for c in list_col if c is not None and not np.isnan(c)]
        st.write(f"--- Distribution for {col} (Total = {len(list_col)})---")
        counts = df_g[col].value_counts()
        pct = df_g[col].value_counts(normalize=True) * 100
        st.write(pd.DataFrame({'Count': counts, 'Percentage (%)': pct}))

        
    

