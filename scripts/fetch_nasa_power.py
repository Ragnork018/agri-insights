#!/usr/bin/env python3
"""Fetch one NASA POWER daily Agroclimatology point series to CSV."""
import argparse, json, urllib.parse, urllib.request
from pathlib import Path
import pandas as pd

PARAMS='PRECTORCORR,T2M,T2M_MAX,T2M_MIN,RH2M'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lat',type=float,required=True);p.add_argument('--lon',type=float,required=True)
    p.add_argument('--start',required=True,help='YYYYMMDD');p.add_argument('--end',required=True,help='YYYYMMDD')
    p.add_argument('--output',type=Path,default=Path('data/weather/nasa_power.csv'))
    args=p.parse_args()
    if not (-90<=args.lat<=90 and -180<=args.lon<=180): p.error('latitude/longitude outside valid ranges')
    if len(args.start)!=8 or len(args.end)!=8 or args.start>args.end: p.error('start/end must be ordered YYYYMMDD')
    q=urllib.parse.urlencode({'parameters':PARAMS,'community':'AG','longitude':args.lon,'latitude':args.lat,'start':args.start,'end':args.end,'format':'JSON','time-standard':'UTC'})
    url='https://power.larc.nasa.gov/api/temporal/daily/point?'+q
    req=urllib.request.Request(url,headers={'User-Agent':'AgriInsights/1.0 (research data client)'})
    with urllib.request.urlopen(req,timeout=60) as response: data=json.load(response)
    params=data.get('properties',{}).get('parameter',{})
    if not params: raise SystemExit('NASA POWER returned no parameter values; check request metadata and date availability')
    frame=pd.DataFrame(params);frame.index.name='date';frame=frame.reset_index()
    args.output.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(args.output,index=False)
    print(f'Wrote {len(frame):,} daily rows to {args.output}; location:({args.lat},{args.lon}); grid metadata={data.get("geometry")}')
if __name__=='__main__':main()
