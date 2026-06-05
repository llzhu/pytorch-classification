import streamlit as st
from ptc_util import *
import plotly.express as px


    
def app_header():
    st.subheader(f'PyTorch Classification in Drug Discovery.')
    read_me_exp = st.expander(f'About PyTorch and Datasets.', expanded=False)
    with read_me_exp:
        st.subheader('PyTorch:')
        st.markdown('PyTorch is a popular open-source machine learning framework used for building and training deep neural networks.')
        st.markdown('https://pytorch.org/')
        
        st.subheader('Tox21 Data Set:')
        st.markdown('The Tox21 dataset is a gold standard for Predictive Toxicology.')
        st.markdown('https://tox21.gov/')
        st.write('***')
    
def app_setup():
    sel1, sel2, sel4 = st. columns([2,2,3])
    with sel1:  # Studies/Datasets
        study = st.selectbox('Pick a dataset', STUDY_OPTIONS)
    with sel2:  # Discriptoes/FP
        X_desc = st.radio('Features:', FEATURE_OPTIONS)
    # with sel3:   # pick a algorithm
    #     algorithm = st.radio(f'Select a Classification Algorithm:', options=MODEL_OPTIONS, horizontal=True)
    with sel4:   # exclusions
        list_to_exclude = st.text_area('Exclude the following in the model training:')
        excluded_list = get_list(list_to_exclude) if list_to_exclude else []
        exclusion_seed = st.text_input("Seed for randomly excluded list (int):", value='42')
        excluded_pct = st.text_input("Percentage for randomly excluded list (int):")
    st.write('***')

    new_or_existing = st.radio('New model or using existing model?', ['Work with an Existing Model', 'Create New Model'], 
                               horizontal=True, disabled=study=='--')
    new_model = new_or_existing == 'Create New Model'
  
    st.write('***')

    return study, X_desc, excluded_list, exclusion_seed, excluded_pct, new_model

def side_data_file_upload(warning_container=None):
    uploaded_data_file = None
    df_upload = None
    uploaded_data_file = st.sidebar.file_uploader("Upload a Data CSV file.")
    st.sidebar.markdown("<small>A SMILES col is required.</small>", unsafe_allow_html=True)
    orig_col_name = expt_col_name = ''
    if uploaded_data_file:
        df_upload = pd.read_csv(uploaded_data_file)
        col_all = df_upload.columns
        col_all = col_all.insert(0, '--')
        if 'SMILES' not in col_all:
            st.stop()

        expt_col = st.sidebar.selectbox('Select Experimental Value Column:', options=col_all)  
        if  expt_col != '--':
           
            if apply_log:
                df_negative = df_upload[df_upload[expt_col]<=0]
                if len(df_negative) > 0:
                    apply_log = False
                    if warning_container:
                        warning_container.warning('Some experimemtal value are not positive. Logarithm cannot be applied')
                    st.stop()

                orig_col_name = expt_col
                expt_col_name = f'log_{expt_col}'
                df_upload[expt_col_name] = df_upload[expt_col].apply(lambda x: math.log10(x))
            else:
                expt_col_name = orig_col_name = expt_col

               
              
        id_col = st.sidebar.selectbox('Select Compund ID Column if available:', options=col_all)  

        out_columns = []

        if id_col and id_col != '--':
            out_columns.append(id_col)

        out_columns.append['SMILES']

        if orig_col_name:
            out_columns.append(orig_col_name)

        if expt_col_name and expt_col_name not in out_columns:
            out_columns.append(expt_col_name)
        
        df_g = df_upload[out_columns]

        if id_col and id_col != '--':
            df_g = df_g.rename(columns={id_col:COMPOUND_ID}) 

        return df_g, expt_col_name




def fig_df_structure(df, expt_label, pred_label, df_container, mol_container, highlight_only):

    fig = px.scatter(
        df,
        x=expt_label,
        y=pred_label,
        custom_data=["row_id"] 
    )

    min_val = min(df[expt_label].min(), df[pred_label].min())
    max_val = max(df[expt_label].max(), df[pred_label].max())

    fig.update_layout(
        shapes=[
                dict(type="rect",
                    xref="paper",
                    yref="paper",
                    x0=0,
                    y0=0,
                    x1=1,
                    y1=1,
                    line=dict(color="black", width=2),
                    fillcolor="rgba(0,0,0,0)"
                )
            ],
        xaxis=dict(range=[min_val, max_val]),
        yaxis=dict(range=[min_val, max_val])
    )

    # Add diagonal x=y line
    fig.add_shape(
        type='line',
        x0=min_val, y0=min_val,
        x1=max_val, y1=max_val,
        line=dict(color='Red', dash='dash'),
        layer='below' # Keeps the line behind the data points
    )

    event = st.plotly_chart(
        fig,
        on_select="rerun",
        width='stretch'
    )

    if event and event.selection and event.selection["points"]:
        selected_ids = [
            point["customdata"]['0'] for point in event.selection["points"]
        ]

    
        def highlight_row(row):
            if row.name in selected_ids:
                return ['background-color: yellow'] * len(row)
            else:
                return [''] * len(row)

        if highlight_only:
            df = df[df["row_id"].isin(selected_ids)]

        style_df = df.style.apply(highlight_row, axis=1)
        df_container.dataframe(style_df, hide_index=True)

        
        smi = df.at[selected_ids[0], SMILES]
        mol = Chem.MolFromSmiles(smi)
        mol_container.write('Selected mol:')
        mol_container.write( moltosvg(mol), unsafe_allow_html=True) 

    else:
        df_container.dataframe(df, hide_index=True)