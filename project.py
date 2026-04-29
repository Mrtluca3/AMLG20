# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: G20
#     language: python
#     name: g20kernel
# ---

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 31, "status": "ok", "timestamp": 1777503511299, "user": {"displayName": "Luca MARTINI", "userId": "16799133474698479044"}, "user_tz": -120} id="e6b6d665" outputId="22fc9ece-761c-424c-e1a2-0d1426975fab"
import torch
import numpy as np
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
print(torch.__version__)

# %% [markdown] id="pG85vCDOJYqq"
# # 1.Preprocessing

# %% [markdown] id="QzUYSkleVXEI"
# ## Load of Dataset via Drive

# %% colab={"base_uri": "https://localhost:8080/", "height": 330} id="YvU6ybx8YMkp" executionInfo={"status": "ok", "timestamp": 1777503620268, "user_tz": -120, "elapsed": 107568, "user": {"displayName": "Luca MARTINI", "userId": "16799133474698479044"}} outputId="d1a19449-0975-4388-8068-493c006114d1"
import gdown
fn_back=r"events_LHCO2020_backgroundMC_Pythia.h5"
url = "https://drive.google.com/uc?id=13ToB6s9MlDtDz2wbt2TxPxcNxzDrmTep"
gdown.download(url, "events_LHCO2020_backgroundMC_Pythia.h5", quiet=False)

fn_signal1= r"events_LHCO2020_BlackBox1.h5"
url_signal1="https://drive.google.com/uc?export=download&id=1jeAtx6V7R-VfDQMwgQUt5GCbPjDbVc8w"
gdown.download(url_signal1, fn_signal1, quiet=False)

fn_signal2= r"events_LHCO2020_BlackBox3.h5"
url_signal2="https://drive.google.com/uc?export=download&id=1vOwsAvnZTC6XFCLbvUBwJWRyAtNZSLMx"
gdown.download(url_signal2, fn_signal2, quiet=False)

# %% id="VQvQp1hCN5dE" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1777503622610, "user_tz": -120, "elapsed": 790, "user": {"displayName": "Luca MARTINI", "userId": "16799133474698479044"}} outputId="6d30770e-8afb-447c-cbfc-6929d70e7a62"
df_background = pd.read_hdf(fn_back,stop=10000)
print(df_background.shape)
print(df_background.head())

# %% colab={"base_uri": "https://localhost:8080/"} id="an6dgZ94Zvnp" executionInfo={"status": "ok", "timestamp": 1777503621264, "user_tz": -120, "elapsed": 987, "user": {"displayName": "Luca MARTINI", "userId": "16799133474698479044"}} outputId="e7ae18fc-f516-450f-cb27-657bddf1e718"
df_signal1 = pd.read_hdf(fn_signal1,stop=10000)
print(df_signal1.shape)
print(df_signal1.head())

# %% colab={"base_uri": "https://localhost:8080/"} id="Wg3q-VYSahdg" executionInfo={"status": "ok", "timestamp": 1777503621815, "user_tz": -120, "elapsed": 548, "user": {"displayName": "Luca MARTINI", "userId": "16799133474698479044"}} outputId="90da3b02-d78e-43df-ee8e-1700c1b157ca"
df_signal2 = pd.read_hdf(fn_signal2,stop=10000)
print(df_signal2.shape)
print(df_signal2.head())
