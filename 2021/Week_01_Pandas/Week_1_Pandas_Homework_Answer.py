import numpy as np
import pandas as pd


df=pd.read_csv('data.csv',parse_dates=True,index_col=[0,1])
print(df.head(10))
'''
Q1
'''
print("q1")
df=df.unstack().KWH

print(df.head(10))

df.fillna(np.nan,inplace=True)
print("q1===================")
;;;
'''
Q2
'''

print("Q2")
print(df.head(10))

for col in range(df.shape[1]):
    df.iloc[:,col][(df.iloc[:,col]-df.mean(axis=1)).abs()>df.std(axis=1)*3]=np.nan
    
'''
Q3
'''
df_max=df.max(axis=1)
df_min=df.min(axis=1)
df_mean=df.mean(axis=1)
df_median=df.median(axis=1)
df_sum=df.sum(axis=1)
df_var=df.var(axis=1)
df_skew=df.skew(axis=1)
df_kurt=df.kurt(axis=1)

df.agg(["max","min","mean","median","sum","std","skew","kurt"])
'''
Q4
'''
df_diff=df.diff(axis=1)

df_diff_max=df_diff.max(axis=1)
df_diff_min=df_diff.min(axis=1)
df_diff_mean=df_diff.mean(axis=1)
df_diff_median=df_diff.median(axis=1)
df_diff_sum=df_diff.sum(axis=1)
df_diff_var=df_diff.var(axis=1)
df_diff_skew=df_diff.skew(axis=1)
df_diff_kurt=df_diff.kurt(axis=1)
df_diff.agg(["max","min","mean","median","sum","std","skew","kurt"])
'''
Q5
'''
df_quantile=df.quantile([.05],axis=1)
print("Q5")
print(df_quantile)
'''
Q6
'''

df_week_diff=df.groupby(df.columns.week,axis=1).sum().diff(axis=1)
print("Q6")
print(df_week_diff.head(10))

df_week_diff_max=df_week_diff.max(axis=1)
df_week_diff_min=df_week_diff.min(axis=1)
df_week_diff_mean=df_week_diff.mean(axis=1)
df_week_diff_median=df_week_diff.median(axis=1)
df_week_diff_sum=df_week_diff.sum(axis=1)
df_week_diff_var=df_week_diff.var(axis=1)
df_week_diff_skew=df_week_diff.skew(axis=1)
df_week_diff_kurt=df_week_diff.kurt(axis=1)
df_week_diff.agg(["max","min","mean","median","sum","std","skew","kurt"])

'''
Q7
'''

df_per10=pd.Series([(df.iloc[row,:]>(df_max*0.9).iloc[row]).sum() for row in range(df.shape[0])],index=range(1,201))
print("Q7")
print(df_per10)

'''
Q8
'''
elec78=df.groupby(df.columns.month,axis=1).sum().iloc[:,6:8]
elec34=df.groupby(df.columns.month,axis=1).sum().iloc[:,2:4]

df_elec_sum_scale=elec78.sum(axis=1)/elec34.sum(axis=1)
df_elec_max_scale=elec78.max(axis=1)/elec34.max(axis=1)
df_elec_min_scale=elec78.min(axis=1)/elec34.min(axis=1)
df_elec_mean_scale=elec78.mean(axis=1)/elec34.mean(axis=1)

print("Q8")
print("df_elec_sum_scale",df_elec_sum_scale.head())
print("df_elec_max_scale",df_elec_max_scale.head())
print("df_elec_min_scale",df_elec_min_scale.head())
print("df_elec_mean_scale",df_elec_mean_scale.head())

'''
Q9
'''

total=pd.DataFrame([df_max,df_diff_max,df_week_diff_max],index=["df_max","df_diff_max","df_week_diff_max"a])
print("Q9")
print(total)