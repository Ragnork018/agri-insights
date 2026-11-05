import streamlit as st, pandas as pd, matplotlib.pyplot as plt
from agri_insights.yield_pipeline import train_yield
st.set_page_config(page_title="Agri Insights · Yield Lab",page_icon="🌾",layout="wide")
st.title("Agri Insights / Yield Lab")
st.caption("District crop-yield research · chronological evaluation")
st.warning("This predicts district-level reported yield, not individual-field outcomes. Confirm the source yield units before reading metrics.")
f=st.file_uploader("Upload DES-style APY CSV",type=["csv"])
if f:
 d=pd.read_csv(f);st.dataframe(d.head(15),use_container_width=True)
 model=st.selectbox("Model",["extra_trees","random_forest","hist_gradient"]);years=st.slider("Hold out latest years",1,5,2)
 if st.button("Train chronological evaluation",type="primary"):
  try: result=train_yield(d,model_name=model,test_years=years);st.session_state["yield_result"]=result
  except Exception as exc: st.error(str(exc))
if "yield_result" in st.session_state:
 r=st.session_state["yield_result"];m=r.metrics;a,b,c=st.columns(3);a.metric("MAE",f"{m['mae']:.2f}");b.metric("RMSE",f"{m["rmse"]:.2f}");c.metric("R2",f"{m["r2"].3f}" if m["r2"] is not None else "n/a");st.json(m)
 fig,ax=plt.subplots();ax.scatter(r.test_rows["yield"],r.predictions,alpha=.5);ax.set_xlabel("Observed yield");ax.set_ylabel("Predicted yield");ax.set_title("Latest-year holdout");st.pyplot(fig)
