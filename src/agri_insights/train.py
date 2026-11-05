import argparse,pandas as pd
from .pipeline import train,save_result

def main():
 p=argparse.ArgumentParser();p.add_argument('--csv',required=True);p.add_argument('--target',required=True);p.add_argument('--algorithm',choices=['random_forest','extra_trees','logistic','xgboost'],default='random_forest');p.add_argument('--out',default='artifacts');a=p.parse_args()
 df=pd.read_csv(a.csv);r=train(df,a.target,a.algorithm);save_result(r,a.out);print(r.metrics)
if __name__=='__main__':main()
