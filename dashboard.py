import streamlit as st,pandas as pd,matplotlib.pyplot as plt
from agri_insights.pipeline import train
st.set_page_config(page_title='Agri Insights',page_icon='âŒ',layout='wide')
st.title('Agri Insights');st.caption('Field performance â· model evaluation workbench')
st.warning('Use a documented field-performance target. This tool does not assume that an arbitrary crop-yield column is equivalent to your resume project label.')
f=st.file_uploader('Upload a CSV dataset',type=['csv'])
if f:
 d=pd.read_csv(f);c1,c2,c3=st.columns(3);c1.metric('Rows',f'{len(d):}');c2.metric('Columns',d.shape[1]);c3.metric('Missing cells',f'{int(d.isna().sum().sum()):}')
st.dataframe(d.head(20),use_container_width=True)
target=st.selectbox('Target label',d.columns);algo=st.selectbox('Model',['random_forest','extra_trees','logistic','xgboost'])
if st.button('Train and evaluate',type='primary'):
¶¼«¶¶¢ÛZ®­jX(²Û²È¨ËZµæ ®*Ş²ém®g­®',­ì\z›DÅÇ©¶*'jÇ¬µêë¢»-­èŸj
â­ë.–Ø§²Û²È¨ËZµé«²Û²È¨ËZµæ ®*Ş²émi·