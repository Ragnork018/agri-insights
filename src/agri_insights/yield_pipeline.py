from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

@dataclass
class YieldResult:
    metrics: dict
    model: object
    test_rows: pd.DataFrame
    predictions: list[float]

ALIASES = {
    'crop_year': ('crop_year','year','crop year'), 'state': ('state','state_name'),
    'district': ('district','district_name'), 'crop': ('crop','crop_name'),
    'season': ('season','crop_season'), 'area': ('area','area_ha','area (ha)'),
    'yield': ('yield','yield_kg_ha','yield (kg/ha)')
}

def _find_column(df, canonical):
    lookup={str(c).strip().lower():c for c in df.columns}
    for alias in ALIASES[canonical]:
        if alias in lookup: return lookup[alias]
    return None

def normalize_apy(df: pd.DataFrame) -> pd.DataFrame:
    """Map common DES APY column names to a canonical district-year schema.

    Yield units are intentionally not converted. They must be verified from the
    downloaded file's data dictionary before comparing metrics across sources.
    """
    out=pd.DataFrame(index=df.index)
    for name in ('crop_year','state','district','crop','season','area','yield'):
        col=_find_column(df,name)
        if col is None: raise ValueError(f'Missing required APY field: {name}; inspect source header and data dictionary')
        out[name]=df[col]
    out['crop_year']=pd.to_numeric(out['crop_year'].astype(str).str.extract(r'(\d{4})')[0],errors='coerce')
    for col in ('area','yield'): out[col]=pd.to_numeric(out[col].astype(str).str.replace(',','',regex=False),errors='coerce')
    out=out.dropna(subset=['crop_year','state','district','crop','season','yield'])
    out=out[(out['yield']>=0)&(out['area']>0)]
    out['crop_year']=out['crop_year'].astype(int)
    if out.empty: raise ValueError('No valid APY records after cleaning')
    return out

def build_yield_features(df: pd.DataFrame) -> pd.DataFrame:
    d=normalize_apy(df)
    d=d.sort_values(['state','district','crop','season','crop_year']).reset_index(drop=True)
    groups=['state','district','crop','season']
    g=d.groupby(groups,dropna=False)['yield']
    d['yield_lag_1']=g.shift(1)
    d['yield_lag_2']=g.shift(2)
    d['yield_mean_prev_3']=g.transform(lambda s:s.shift(1).rolling(3,min_periods=1).mean())
    return d

def train_yield(df: pd.DataFrame, model_name='extra_trees', test_years=2, seed=42) -> YieldResult:
    d=build_yield_features(df)
    years=sorted(d['crop_year'].dropna().unique())
    if len(years)<4: raise ValueError('At least four crop years are required for a chronological holdout')
    test_years=max(1,min(int(test_years),len(years)-2)); cutoff=years[-test_years]
    train=d[d.crop_year<cutoff].copy(); test=d[d.crop_year>=cutoff].copy()
    if len(train)<30 or len(test)<5: raise ValueError('Not enough rows across the chosen time split; inspect dataset coverage')
    target='yield'; forbidden={target,'production','production_tonnes','yield_current_year'}
    numeric=['crop_year','area','yield_lag_1','yield_lag_2','yield_mean_prev_3']
    categorical=['state','district','crop','season']
    features=numeric+categorical
    Xtr=train[features].copy(); Xte=test[features].copy()
    # Keep a transparent previous-year baseline for records where lag_1 exists.
    baseline=test['yield_lag_1'].copy()
    tx=[('num',SimpleImputer(strategy='median',add_indicator=True),numeric),
        ('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',min_frequency=2))]),categorical)]
    prep=ColumnTransformer(tx)
    if model_name=='random_forest': model=RandomForestRegressor(n_estimators=400,min_samples_leaf=2,max_features=0.9,n_jobs=-1,random_state=seed)
    elif model_name=='hist_gradient': model=HistGradientBoostingRegressor(max_iter=300,learning_rate=.05,l2_regularization=1.0,random_state=seed)
    elif model_name=='extra_trees': model=ExtraTreesRegressor(n_estimators=500,min_samples_leaf=2,max_features=0.9,n_jobs=-1,random_state=seed)
    else: raise ValueError('model_name must be extra_trees, random_forest, or hist_gradient')
    pipe=Pipeline([('preprocess',prep),('model',model)])
    pipe.fit(Xtr,train[target]); pred=np.maximum(0,pipe.predict(Xte))
    baseline_mask=baseline.notna()
    metrics={'rows':len(d),'train_rows':len(train),'test_rows':len(test),'train_year_max':int(train.crop_year.max()),'test_year_min':int(test.crop_year.min()),'test_year_max':int(test.crop_year.max()),'model':model_name,'features':features,'excluded_same_year_outcomes':sorted(forbidden),'mae':float(mean_absolute_error(test[target],pred)),'rmse':float(np.sqrt(mean_squared_error(test[target],pred))),'r2':float(r2_score(test[target],pred)) if len(test)>1 else None,'previous_year_baseline_mae':float(mean_absolute_error(test.loc[baseline_mask,target],baseline[baseline_mask])) if baseline_mask.any() else None,'note':'Temporal district/crop holdout; verify APY yield units before interpreting metrics.'}
    return YieldResult(metrics,pipe,test.reset_index(drop=True),pred.tolist())
