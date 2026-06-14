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

    
if not app_vars.is_admin:
    st.write('Only admins can copy models to Master folders.')
    st.stop()

copy_to_master = False
c1, c2 = st.columns(2)
with c1:
    sel_models = ['My model', 'Master Model']
    loaded_model = st.selectbox(f'Load a saved model:', sel_models)
with c2:
    copy_container = st.container()

if loaded_model == 'My model':  
    prefix = get_prefix(env, app_vars, model_desc)
    folder_name = app_vars.login_name

    copy_to_master = copy_container.checkbox('Copy model to Master folder')
   
else:
    prefix = get_prefix_master(env, app_vars, model_desc)
    folder_name = 'master'

is_any_contents = any_contents(env.s3_bucket, prefix)

if not is_any_contents:
    st.write(f"""No {model_desc.model_class} model with {model_desc.X_desc} features are available 
                 for {app_vars.study} in {folder_name} folder""")
    st.stop()

model_key = f'{prefix}model_desc.pkl'
model_desc:ModelDesc = get_from_s3(env.s3_bucket, model_key)
# app_key =  f'{prefix}app_vars.pkl'
# app_vars:AppVars = get_from_s3(env.s3_bucket, app_key)

# model_class = model_desc.model_class
# model = model_desc.model
# classes = app_vars.classes

st.write(model_desc.model_class)
st.write(model_desc.model)

if copy_to_master:
    
    master_prefix = get_prefix_master(env, app_vars, model_desc)
    copy_s3_folder(env.s3_bucket, prefix, master_prefix)

    copy_container.write('Model has been copied to Master folder.')
