# -*- coding: utf-8 -*-
"""
Created on Sat Mar 04 22:36:44 2017

@author: AshwinthPS - First Submission - Seed diff model - 0.572604 logloss
"""

import numpy as np 
import pandas as pd 
import os

mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-5.3.0-posix-seh-rt_v4-rev0\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import xgboost as xgb

data_dir = 'C:\Users\HP\Desktop\Hackathon\March Machine Learning Mania'
df_seeds = pd.read_csv(data_dir + '\TourneySeeds.csv')
df_tour = pd.read_csv(data_dir + '\TourneyCompactResults.csv')
df_tourd=pd.read_csv(data_dir+'\TourneyDetailedResults.csv')
df_sub =pd.read_csv(data_dir+ "\sample_submission.csv")
df_regd=pd.read_csv(data_dir+'\RegularSeasonDetailedResults.csv')

###################################################  Seeds    ########################################

def seed_to_int(seed):
    """Get just the digits from the seeding. Return as int"""
    s_int = int(seed[1:3])
    return s_int
    
df_seeds['Seeds']=df_seeds.Seed.apply(seed_to_int)
df_seeds.head()
df_seeds=df_seeds.drop('Seed',axis=1)
df_seeds.columns=['Season','team1','Seeds']

df_seeds2=df_seeds.copy()
df_seeds2.columns=['Season','team2','Seeds']

#########################################################################################################

final_df=pd.DataFrame()
final_df[["Season","team1","team2"]] = df_tour[["Season","Wteam","Lteam"]].copy()
final_df['pred']=1
final_df = pd.merge(left=final_df, right=df_seeds, how='left', on=['Season', 'team1'])
final_df = pd.merge(left=final_df, right=df_seeds2, how='left', on=['Season', 'team2'])
final_df['seed_dif']=final_df['Seeds_x']-final_df['Seeds_y']

final_df2 = pd.DataFrame()
final_df2[["Season","team1", "team2"]] =df_tour[["Season","Lteam", "Wteam"]].copy()
final_df2["pred"] = 0
final_df2 = pd.merge(left=final_df2, right=df_seeds, how='left', on=['Season', 'team1'])
final_df2 = pd.merge(left=final_df2, right=df_seeds2, how='left', on=['Season', 'team2'])
final_df2['seed_dif']=final_df2['Seeds_x']-final_df2['Seeds_y']
final_df2.head()

final = pd.concat((final_df,final_df2), axis=0)
final.iloc[2050:2055,:]

final_pred=final[final['Season'] < 2013]

pred=final_pred.loc[:,['Season',  'team1'  ,'team2' ,'seed_dif' ]]

df_sub.columns=['Season','team1','team2']
df_sub=pd.merge(df_sub,df_seeds,how='left',on=['Season','team1'])
df_sub=pd.merge(df_sub,df_seeds2,how='left',on=['Season','team2'])
df_sub['seed_dif']=df_sub['Seeds_x']-df_sub['Seeds_y']
df_sub=df_sub.drop(['Seeds_x','Seeds_y'],axis=1)

###########################################################  MODEL #############################################
params = {"objective": "binary:logistic","booster": "gbtree", "nthread": 4, "silent": 1,'eval_metric':'logloss',
                "eta": 0.1, "max_depth": 3, "subsample": 0.9, "colsample_bytree": 0.7, 
                "min_child_weight": 5,"seed": 2016, "tree_method": "exact"}

params

dtrain = xgb.DMatrix(pred, final_pred.pred, missing=np.nan)

nrounds=50

watchlist = [(dtrain, 'train')]

bst = xgb.train(params, dtrain, num_boost_round=nrounds, evals=watchlist, verbose_eval=20)

dtest = xgb.DMatrix(df_sub)#, missing=np.nan)

test_preds = bst.predict(dtest)

test_preds

op1=pd.DataFrame(test_preds)

op1.to_csv("subm.csv")

################################################################################################################