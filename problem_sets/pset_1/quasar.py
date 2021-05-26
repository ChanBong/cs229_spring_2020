import seaborn as sns
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp

df_train = pd.read_csv('./data/quasar_train.csv', )
cols_train = df_train.columns.values.astype(float).astype(int)
df_train.shape
df_train.head()
