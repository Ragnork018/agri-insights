import pandas as pd,pytest
from agri_insights.pipeline import train,validate_frame

def sample():
 rows=[]
 for i in range(120):rows.append({'soil_ph':5.5+(i%5)*.3,'rain_mm':100+(i%7)*12,'crop':['rice','maize','wheat'][i%3],'performance':['low','medium','high'][i%3],'id':i})
 return pd.DataFrame(rows)
def test_train_produces_valid_holdout_metrics():
 r=train(sample(),'performance');assert 0<=r.metrics['macro_f1']<=1;assert r.metrics['test_rows']>0;assert 'id' in r.metrics['dropped_columns']
def test_missing_target_rejected():
 d=sample();d.loc[0,'performance']=None
 with pytest.raises(ValueError):train(d,'performance')
