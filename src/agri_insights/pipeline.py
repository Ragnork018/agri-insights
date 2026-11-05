from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json, joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

@dataclass
class TrainingResult:
    metrics:dict; report:dict; confusion:list; labels:list; model:object; feature_columns:list

def validate_frame(df:pd.DataFrame,target:str):
    if target not in df:raise ValueError(f"Target column '{target}' is missing")
    if len(df)<30:raise ValueError('At least 30 rows are required for a minimally useful holdout')
    if df[target].isna().any():raise ValueError('Target contains missing labels; resolve them explicitly before training')
    counts=df[target].value_counts()
    if len(counts)<2:raise ValueError('Target must contain at least two classes')
    if counts.min()<2:raise ValueError('Each class needs at least two rows for a stratified split')
    if df[target].nunique()>50:raise ValueError('More than 50 target classes; verify that target is not an ID or continuous measurement')

def build_pipeline(X,algorithm='random_forest'):
    numeric=list(X.select_dtypes(include=['number','bool']).columns)
    categorical=[c for c in X.columns if c not in numeric]
    transformers=[]
    if numeric:transformers.append(('numeric',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]),numeric))
    if categorical:transformers.append(('categorical',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',min_frequency=2))]),categorical))
    prep=ColumnTransforer(transformers,remainder='drop')
    if algorithm=='logistic':model=LogisticRegression(max_iter=1500,class_weight='balanced')
    elif algorithm=='xgboost':
        try: from xgboost import XGBClassifier
        except ImportError as exc: raise ValueError("Install the optional XGBoost extra: pip install -e '.[xgboost]'") from exc
        model=XGBClassifier(n_estimators=450,max_depth=5,learning_rate=.04,subsample=.9,colsample_bytree=.9,reg_lambda=2,eval_metric='mlogloss',tree_method='hist',random_state=42,n_jobs=-1)
    elif algorithm=='extra_trees':model=ExtraTreesClassifier(n_estimators=350,min_samples_leaf=2,class_weight='balanced', random_state=42,n_jobs=-1)
    else:model=RandomForestClassifier(n_estimators=350,min_samples_leaf=2,class_weight='balanced_subsample',random_state=42,n_jobs=-1)
    return Pipeline([('preprocess',prep),('model',model)])

def train(df:pd.DataFrame,target:str,algorithm='random_forest',test_size=.2,seed=42)->TrainingResult:
    validate_frame(df,target)
    # Remove constant columns and obvious row identifiers; keep this list visible for audit.
    X=df.drop(columns=[target]).copy(); y_text=df[target].astype(str)
    label_encoder=LabelEncoder(); y=label_encoder.fit_transform(y_text)
    drop=[c for c in X.columns if X[c].nunique(dropna=False)<=1 or c.lower() in {'id','row_id','record_id'}]
    X=X.drop(columns=drop)
    if X.shape[1]==0:raise ValueError('No usable feature columns remain after removing target and constants/IDs')
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test_size,random_state=seed,stratify=y)
    pipe=build_pipeline(Xtr,algorithm);pipe.fit(Xtr,ytr);pred=pipe.predict(Xte)
    labels=list(range(len(label_encoder.classes_)));pred_names=label_encoder.inverse_transform(pred);true_names=label_encoder.inverse_transform(yte)
    cm=confusion_matrix(yte,pred,labels=labels)
    metrics={'rows':len(df),'train_rows':len(Xtr),'test_rows':len(Xte),'features':X.shape[1],'classes':len(labels),'class_names':label_encoder.classes_.tolist(),'algorithm':algorithm,'accuracy':float(accuracy_score(yte,pred)),'balanced_accuracy':float(balanced_accuracy_score(yte,pred)),'macro_f1':float(f1_score(yte,pred,average='macro',zero_division=0)),'dropped_columns':drop,'seed':seed}
    report=classification_report(true_names,pred_names,output_dict=True,zero_division=0)
    return TrainingResult(metrics,report,cm.tolist(),label_encoder.classes_.tolist(),pipe,list(X.columns))

def save_result(result:TrainingResult,outdir):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    joblib.dump(result.model,out/'model.joblib')
    (out/'metrics.json').write_text(json.dumps(result.metrics,indent=2))
    (out/'classification_report.json').write_text(json.dumps(result.report,indent=2))
    pd.DataFrame(result.confusion,index=result.labels,columns=result.labels).to_csv(out/'confusion_matrix.csv')
    (out/'model_card.md').write_text('# Model card\n\nThis model was trained on a user-supplied dataset. Review `metrics.json` for row counts, split seed and measured metrics. This benchmark is not a causal agronomic recommendation. Validate external generalization by location, season and crop before operational use.\n')
