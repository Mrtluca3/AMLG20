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

# %% id="e6b6d665" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1777495289213, "user_tz": -120, "elapsed": 5448, "user": {"displayName": "Luca MARTINI", "userId": "16799133474698479044"}} outputId="adb408e0-887f-48c8-bb5a-0a94468958f5"
import torch
import numpy as np
from torch import nn
print(torch.__version__)

# %% [markdown] id="pG85vCDOJYqq"
# # Preprocessing

# %% id="cBTEeaKMOCT3"
fn_signal= r"file.h5"
fn_back=r"back."

# %% id="VQvQp1hCN5dE"
import pandas as pd
df_test = pd.read_hdf(fn_back,stop=10000)
print(df_test.shape)
print("Memory in GB:",sum(df_test.memory_usage(deep=True)) / (1024**3))

print(df_test.head())
print(df_test.shape)
