# -*- coding: utf-8 -*-
# Brazil DI Futures PCA — stable forecast + robust in-sample reconstruction

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Brazil DI PCA — Forecast & Tools")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120

# --------------------------
# Helpers
# --------------------------
def safe_to_datetime(s):
    return pd.to_datetime(s, errors="coerce", dayfirst=False)

def normalize_rate_input(val, unit):
    if pd.isna(val): return np.nan
    v = float(val)
    if "Percent" in unit: return v / 100.0
    if "Basis" in unit: return v / 10000.0
    return v

def denormalize_to_percent(frac):
    if pd.isna(frac): return np.nan
    return 100.0 * float(frac)

def np_busdays_exclusive(start_dt, end_dt, holidays_np):
    if pd.isna(start_dt) or pd.isna(end_dt): return 0
    s = np.datetime64(pd.Timestamp(start_dt).date()) + np.timedelta64(1,"D")
    e = np.datetime64(pd.Timestamp(end_dt).date())
    if e < s: return 0
    return int(np.busday_count(s,e,weekmask="1111100",holidays=holidays_np))

def calculate_ttm(valuation_ts, expiry_ts, holidays_np, year_basis):
    bd = np_busdays_exclusive(valuation_ts,expiry_ts,holidays_np)
    return np.nan if bd <= 0 else bd/float(year_basis)

def next_business_day(date_like, holidays_np):
    d = np.datetime64(pd.Timestamp(date_like).date())
    nxt = np.busday_offset(d,1,weekmask="1111100",holidays=holidays_np)
    return pd.Timestamp(nxt)

def build_std_grid_by_rule(max_year=7.0):
    a = list(np.round(np.arange(0.25,3.0+0.001,0.25),2))
    b = list(np.round(np.arange(3.5,5.0+0.001,0.5),2))
    c = list(np.round(np.arange(6.0,max_year+0.001,1.0),2))
    return a+b+c

# --------------------------
# Sidebar
# --------------------------
st.sidebar.header("Upload / Settings")

yield_file = st.sidebar.file_uploader("1) Yield data CSV", type="csv")
expiry_file = st.sidebar.file_uploader("2) Expiry mapping CSV", type="csv")
holiday_file = st.sidebar.file_uploader("3) Holiday dates CSV (optional)", type="csv")

interp_method = st.sidebar.selectbox("Interpolation method",["linear","cubic","quadratic","nearest"])
apply_smoothing = st.sidebar.checkbox("Apply smoothing (3-day centered)", value=False)
n_components_sel = st.sidebar.slider("Number of PCA components",1,12,3)

rate_unit = st.sidebar.selectbox("Input rate unit",["Percent (e.g. 13.45)","Decimal (e.g. 0.1345)","Basis points (e.g. 1345)"])
year_basis = int(st.sidebar.selectbox("Business days in year",[252,360],index=0))

compounding_model = st.sidebar.radio("Compounding model",["identity (input already annual zero / effective)","DI daily business compounding"],index=0)

use_grid_rule = st.sidebar.checkbox("Use standard grid rule",value=True)
std_maturities_txt = st.sidebar.text_input("Custom standard maturities","0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00,2.25,2.50,2.75,3.00,3.50,4.00,4.50,5.00,6.00,7.00")

st.sidebar.markdown("---")
st.sidebar.header("Forecasting Model")
forecast_model_type = st.sidebar.selectbox("Select Forecasting Model",["None (PCA Fair Value)","Average Delta (Momentum)","VAR (Vector Autoregression)","ARIMA (per Component)"])

if forecast_model_type=="Average Delta (Momentum)":
    rolling_window_days = st.sidebar.number_input("PCs avg-delta window",1,20,5)
    pc_damp = st.sidebar.slider("Damping for PC2..PCn",0.0,1.0,0.5,0.05)
elif forecast_model_type=="VAR (Vector Autoregression)":
    var_lags = st.sidebar.number_input("VAR model lags",1,20,1)
elif forecast_model_type=="ARIMA (per Component)":
    st.sidebar.write("Using ARIMA(1,1,0) for each PC.")

apply_cap = st.sidebar.checkbox("Apply daily cap", value=True)
cap_bps = st.sidebar.number_input("Cap per-day move (bps)",0,500,10)
cap_frac = cap_bps/10000.0

show_previous_curve = st.sidebar.checkbox("Show Previous Day's Curve", value=True)

# --------------------------
# Load Data
# --------------------------
def load_csv_file(f): return pd.read_csv(io.StringIO(f.getvalue().decode("utf-8")))

if yield_file is None or expiry_file is None: st.stop()

yields_df = load_csv_file(yield_file)
date_col = yields_df.columns[0]
yields_df[date_col] = safe_to_datetime(yields_df[date_col])
yields_df = yields_df.dropna(subset=[date_col]).set_index(date_col).sort_index()
for c in yields_df.columns: yields_df[c] = pd.to_numeric(yields_df[c],errors="coerce")

expiry_raw = load_csv_file(expiry_file)
expiry_df = expiry_raw.iloc[:, :2].copy()
expiry_df.columns=["MATURITY","DATE"]
expiry_df["DATE"] = safe_to_datetime(expiry_df["DATE"])
expiry_df = expiry_df.dropna().set_index("MATURITY")

holidays_np = np.array([],dtype="datetime64[D]")
if holiday_file:
    hol_df = load_csv_file(holiday_file)
    hol_series = safe_to_datetime(hol_df.iloc[:,0]).dropna()
    holidays_np = np.array(hol_series.dt.date,dtype="datetime64[D]")

min_date_available = yields_df.index.min().date()
max_date_available = yields_df.index.max().date()
date_train_start = st.sidebar.date_input("Training Start", value=min_date_available,
                                         min_value=min_date_available,max_value=max_date_available)
date_train_end = st.sidebar.date_input("Training End", value=max_date_available,
                                       min_value=min_date_available,max_value=max_date_available)

train_mask = (yields_df.index.date>=date_train_start)&(yields_df.index.date<=date_train_end)
yields_df_train = yields_df.loc[train_mask]
if apply_smoothing: yields_df_train = yields_df_train.rolling(3,min_periods=1,center=True).mean()

if use_grid_rule: std_arr = np.array(build_std_grid_by_rule(7.0))
else: std_arr = np.array(sorted([float(x) for x in std_maturities_txt.split(",") if x.strip()!=""]))
std_cols = [f"{m:.2f}Y" for m in std_arr]

# --------------------------
# Core transformation
# --------------------------
def row_to_std_grid(dt,row_series,contracts,expiry_df,std_arr,holidays_np,year_basis,rate_unit,comp_model,interp_method):
    ttm_list, zero_list = [], []
    for col in contracts:
        if col not in expiry_df.index: continue
        exp = expiry_df.loc[col,"DATE"]
        if pd.isna(exp) or pd.Timestamp(exp).date()<dt.date(): continue
        t = calculate_ttm(dt,exp,holidays_np,year_basis)
        if np.isnan(t) or t<=0: continue
        raw_val=row_series.get(col,np.nan)
        if pd.isna(raw_val): continue
        r_frac = normalize_rate_input(raw_val,rate_unit)
        if comp_model.startswith("identity"): zero_frac=r_frac
        else:
            T=int(np_busdays_exclusive(dt,exp,holidays_np))
            if T==0: continue
            r_daily=r_frac/year_basis
            DF=(1+r_daily)**(-T)
            zero_frac=DF**(-1.0/t)-1.0
        ttm_list.append(t); zero_list.append(denormalize_to_percent(zero_frac))
    if len(ttm_list)>1:
        order=np.argsort(ttm_list)
        f=interp1d(np.array(ttm_list)[order],np.array(zero_list)[order],
                   kind=interp_method,bounds_error=False,fill_value=np.nan)
        return f(std_arr)
    return np.full_like(std_arr,np.nan)

pca_df = pd.DataFrame(np.nan,index=yields_df_train.index,columns=std_cols)
for dt in yields_df_train.index:
    pca_df.loc[dt] = row_to_std_grid(dt,yields_df_train.loc[dt],yields_df_train.columns,
                                     expiry_df,std_arr,holidays_np,year_basis,rate_unit,compounding_model,interp_method)
pca_vals=pca_df.values.astype(float)
col_means=np.nanmean(pca_vals,axis=0)
inds=np.where(np.isnan(pca_vals))
if inds[0].size>0: pca_vals[inds]=np.take(col_means,inds[1])
pca_df_filled=pd.DataFrame(pca_vals,index=pca_df.index,columns=pca_df.columns)

# --------------------------
# PCA Fit on training window
# --------------------------
scaler=StandardScaler(with_std=False)
X=scaler.fit_transform(pca_df_filled.values)
max_components=int(min(X.shape[0],X.shape[1]))
n_components=min(n_components_sel,max_components)
pca=PCA(n_components=n_components)
PCs=pca.fit_transform(X)
PCs_df=pd.DataFrame(PCs,index=pca_df_filled.index,columns=[f"PC{i+1}" for i in range(n_components)])

# --------------------------
# Heatmap
# --------------------------
st.header("PCA Components Heatmap (Training Window)")
fig,ax=plt.subplots(figsize=(12,max(4,n_components*1.5)))
sns.heatmap(pca.components_,xticklabels=std_cols,yticklabels=PCs_df.columns,
            annot=True,fmt=".2f",cmap="viridis",ax=ax)
ax.set_title(f"PCA Component Loadings vs. Maturity\nTraining {date_train_start} → {date_train_end}")
st.pyplot(fig)

# --------------------------
# Reconstruction
# --------------------------
st.header("In-Sample Reconstruction")
recon_min,recon_max=pca_df_filled.index.min().date(),pca_df_filled.index.max().date()
recon_date=st.date_input("Select date",value=recon_max,min_value=recon_min,max_value=recon_max)
recon_ts=pd.Timestamp(recon_date)
if recon_ts not in pca_df_filled.index: st.stop()
i=pca_df_filled.index.get_loc(recon_ts)
recon_curve=scaler.inverse_transform(pca.inverse_transform(PCs[i].reshape(1,-1))).ravel()
orig_curve=pca_df_filled.loc[recon_ts].values
fig,ax=plt.subplots(figsize=(12,6))
ax.plot(std_arr,recon_curve,'-o',label="Reconstructed")
ax.plot(std_arr,orig_curve,'--x',label="Original")
ax.legend(); st.pyplot(fig)

# --------------------------
# Forecast (by contract)
# --------------------------
st.header("Next-BD Forecast")
train_end_ts=pca_df_filled.index.max()
forecast_ts=next_business_day(train_end_ts,holidays_np)

def forecast_pcs_avg_delta(PCs,window=5,pc_damp=0.5):
    n,k=PCs.shape; last=PCs[-1].reshape(1,-1)
    if n<2: return last
    avg_delta=np.mean(np.diff(PCs[-min(window,n):],axis=0),axis=0)
    damp=np.ones(k); damp[1:]=pc_damp
    return last+(avg_delta*damp).reshape(1,-1)

if forecast_model_type=="None (PCA Fair Value)":
    pred_curve=scaler.inverse_transform(pca.inverse_transform(PCs[-1].reshape(1,-1))).ravel()
elif forecast_model_type=="Average Delta (Momentum)":
    pcs_next=forecast_pcs_avg_delta(PCs,rolling_window_days,pc_damp)
    delta_curve=pca.inverse_transform(pcs_next-PCs[-1])
    pred_curve=scaler.inverse_transform(pca.inverse_transform(PCs[-1].reshape(1,-1))).ravel()+delta_curve.flatten()
elif forecast_model_type=="VAR (Vector Autoregression)":
    if len(PCs_df)<var_lags+5: pcs_next=PCs_df.iloc[-1:].values
    else: pcs_next=VAR(PCs_df).fit(var_lags).forecast(PCs_df.values[-var_lags:],steps=1)
    delta_curve=pca.inverse_transform(pcs_next-PCs[-1])
    pred_curve=scaler.inverse_transform(pca.inverse_transform(PCs[-1].reshape(1,-1))).ravel()+delta_curve.flatten()
elif forecast_model_type=="ARIMA (per Component)":
    preds=[]
    for col in PCs_df.columns:
        s=PCs_df[col]
        preds.append(ARIMA(s,order=(1,1,0)).fit().forecast(1).iloc[0] if len(s)>=10 else s.iloc[-1])
    pcs_next=np.array(preds).reshape(1,-1)
    delta_curve=pca.inverse_transform(pcs_next-PCs[-1])
    pred_curve=scaler.inverse_transform(pca.inverse_transform(PCs[-1].reshape(1,-1))).ravel()+delta_curve.flatten()

if apply_cap and forecast_model_type!="None (PCA Fair Value)":
    last_curve=scaler.inverse_transform(pca.inverse_transform(PCs[-1].reshape(1,-1))).ravel()
    delta=np.clip(pred_curve-last_curve,-cap_frac,cap_frac)
    pred_curve=last_curve+delta

# Map back to contracts
def std_grid_to_contracts(dt,std_curve,std_arr,expiry_df,contracts,holidays_np,year_basis,rate_unit,comp_model,interp_method):
    rates=pd.Series(index=contracts,dtype=float)
    f=interp1d(std_arr,std_curve,kind=interp_method,bounds_error=False,fill_value="extrapolate")
    for c in contracts:
        if c not in expiry_df.index: continue
        exp=expiry_df.loc[c,"DATE"]
        if pd.isna(exp) or pd.Timestamp(exp).date()<dt.date(): continue
        t=calculate_ttm(dt,exp,holidays_np,year_basis)
        if np.isnan(t) or t<=0: continue
        zero=f(t)/100.0
        if comp_model.startswith("identity"): rate_frac=zero
        else:
            T=int(np_busdays_exclusive(dt,exp,holidays_np))
            if T==0: continue
            DF=(1+zero)**(-t)
            r_daily=DF**(-1.0/T)-1.0
            rate_frac=r_daily*year_basis
        if "Percent" in rate_unit: rates[c]=rate_frac*100
        elif "Basis" in rate_unit: rates[c]=rate_frac*10000
        else: rates[c]=rate_frac
    return rates.dropna()

all_contracts=yields_df.columns
pred_contracts=std_grid_to_contracts(forecast_ts,pred_curve,std_arr,expiry_df,all_contracts,
                                     holidays_np,year_basis,rate_unit,compounding_model,interp_method)
last_contracts=std_grid_to_contracts(train_end_ts,pca_df_filled.iloc[-1].values,std_arr,expiry_df,all_contracts,
                                     holidays_np,year_basis,rate_unit,compounding_model,interp_method)

fig,ax=plt.subplots(figsize=(12,6))
ax.plot(last_contracts.index,last_contracts.values,'-s',label=f"Last Actual ({train_end_ts.date()})")
ax.plot(pred_contracts.index,pred_contracts.values,'-o',label=f"Predicted ({forecast_ts.date()})")
ax.legend(); st.pyplot(fig)

st.subheader("Forecast Data by Contract")
forecast_df=pd.DataFrame({f"Last Actual ({train_end_ts.date()})":last_contracts,
                          f"Predicted ({forecast_ts.date()})":pred_contracts})
st.dataframe(forecast_df.style.format("{:.2f}"))

# --------------------------
# Fair Value Spread
# --------------------------
st.header("Fair Value Spread Analysis")
fair_curve=scaler.inverse_transform(pca.inverse_transform(PCs[-1].reshape(1,-1))).ravel()
fair_contracts=std_grid_to_contracts(train_end_ts,fair_curve,std_arr,expiry_df,all_contracts,
                                     holidays_np,year_basis,rate_unit,compounding_model,interp_method)
spread=(last_contracts-fair_contracts)*100
fig,ax=plt.subplots(figsize=(12,6))
spread.plot(kind="bar",ax=ax,color=['g' if x<0 else 'r' for x in spread])
ax.axhline(0,color="k"); st.pyplot(fig)
spread_df=pd.DataFrame({"Actual":last_contracts,"Fair":fair_contracts,"Spread(bps)":spread})
st.dataframe(spread_df.style.format("{:.2f}"))
