import pandas as pd
import numpy as np
from itertools import product
import seaborn as sns
import os
import matplotlib.pyplot as plt
import scipy.sparse 

from sklearn.metrics import mean_squared_error,make_scorer
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import lightgbm as lgb
import xgboost as xgb



def get_submission(item_cnt_month,sub_name,clip=20,data_path ='data/' ):
	item_cnt_month = np.clip(item_cnt_month,0,clip)
	test= pd.read_csv(os.path.join(data_path, 'test.csv.gz'))
	sub = test.copy()
	sub['item_cnt_month'] = item_cnt_month
	sub.drop(['item_id','shop_id'],axis=1,inplace=True)
	sub.to_csv(data_path+'submission/' + sub_name+'.csv',index=False)
	return sub
def plot_xgb_feature_importances(importances):
	sorted_columns=[]
	sorted_values=[]
	for key in sorted(importances, key=importances.get,reverse=False):
		sorted_columns.append(key)
		sorted_values.append(importances[key])

	length = len(importances)
	plt.figure(figsize=(10, 10))
	plt.title('Feature Importances')
	plt.barh(range(length),sorted_values[:length],color='lightblue',align='center',height=0.8)
	plt.yticks(range(length),sorted_columns[:length])
	plt.ylim([-1,length])
	plt.tight_layout()

def downcast_dtypes(df):
	'''
	Changes column types in the dataframe: 
		
		`float64` type to `float32`
		`int64`   type to `int32`
	'''

	# Select columns to downcast
	float_cols = [c for c in df if df[c].dtype == "float64"]
	int_cols =   [c for c in df if df[c].dtype == "int64"]

	# Downcast
	df[float_cols] = df[float_cols].astype(np.float32)
	df[int_cols]   = df[int_cols].astype(np.int32)

	return df

def get_cv_idxs(df,start,end):
	result=[]
	for i in range(start,end+1):
		dates = df.date_block_num
		train_idx = np.array(df.loc[dates <i].index)
		val_idx = np.array(df.loc[dates == i].index)
		result.append((train_idx,val_idx))
	return np.array(result)
def get_X_y(df,end,clip=20):
	# don't drop date_block_num
	df = df.loc[df.date_block_num <= end]
	cols_to_drop=['target','item_name'] + df.columns.values[6:12].tolist()
	y = np.clip(df.target.values,0,clip)
	X = df.drop(cols_to_drop,axis=1)
	return X,y

def root_mean_squared_error(truth,pred):
	return sqrt(mean_squared_error(truth,pred))

def get_all_data(data_path,filename):
	all_data = pd.read_pickle(data_path + filename)
	all_data = downcast_dtypes(all_data)
	all_data = all_data.reset_index().drop('index',axis=1)
	return all_data
def get_all_data_sample(data_path,filename,ratio=.2,seed=42):
	all_data = get_all_data(data_path,filename)
	n_sample = int(all_data.shape[0] * ratio)
	np.random.seed(seed)
	idx_sample = np.random.choice(all_data.shape[0], n_sample, replace=False)
	all_data_sample = all_data.iloc[idx_sample].copy()
	all_data_sample = all_data_sample.reset_index().drop('index',axis=1)
	return all_data_sample
    
def get_train_val(X,y,val_block):
	if val_block>33:
		raise ValueError('Maximum date_block_n is 33')
	X_train  = X[X.date_block_num<val_block].copy()
	X_val = X[X.date_block_num==val_block].copy()
	y_train = y[X_train.index.tolist()].copy()
	y_val = y[X_val.index.tolist()].copy()
	X_train.drop('date_block_num',axis=1,inplace=True)
	X_val.drop('date_block_num',axis=1,inplace=True)
	return X_train,X_val,y_train,y_val

def timeseries_cv(clf_name,X,y,params,cv,loss_metric,early_stopping_round=100,get_oof=False,extra_rounds=1):
	'''
	Doing XGBoost and LightGBM CV for time series.
	clf_name: 'xgb' or 'lgb'
	cv: [(train idx time 1,val idx time 1),( train idx time 2, val idx time 2), ...]
	'''
	print("Training with params: ")
	print(params)

	oof_train = np.zeros([0,])
	cv_losses=[]
	cv_iteration=[]

	for (train_idx,val_idx) in cv:
		cv_train = X.iloc[train_idx]
		cv_val = X.iloc[val_idx]
		cv_y_train = y[train_idx]
		cv_y_val = y[val_idx]

		train_pred=None
		val_pred=None
		best_nround=0
		if clf_name == 'lgb':
            
			lgb_model = lgb.train(params, lgb.Dataset(cv_train, label=cv_y_train), 2000, 
				      lgb.Dataset(cv_val, label=cv_y_val), verbose_eval=False, 
				      early_stopping_rounds=early_stopping_round)
			best_nround=lgb_model.best_iteration
			train_pred = lgb_model.predict(cv_train,best_nround)
			val_pred = lgb_model.predict(cv_val,best_nround+extra_rounds)
        

		elif clf_name == 'xgb':
			dtrain = xgb.DMatrix(cv_train,cv_y_train)
			dval = xgb.DMatrix(cv_val,cv_y_val)
			watchlist = [(dtrain, 'train'), (dval, 'valid')]
			xgb_model = xgb.train(params, dtrain, 2000, watchlist,
				      verbose_eval=False, 
				      early_stopping_rounds=early_stopping_round)
			best_nround=xgb_model.best_ntree_limit
			train_pred = xgb_model.predict(dtrain,ntree_limit=best_nround)
			val_pred = xgb_model.predict(dval,ntree_limit=best_nround+extra_rounds)

			xgb_model.__del__()
		else:
			return None
        
        #  oof_train[dbn_level2==current_bn] = val_pred
		if get_oof:
			oof_train = np.append(oof_train,val_pred)
    
		val_loss = loss_metric(cv_y_val,val_pred)
		train_loss = loss_metric(cv_y_train,train_pred)
		print('Train RMSE: {}. Val RMSE: {}'.format(train_loss,val_loss))
		print('Best iteration: {}'.format(best_nround))
		cv_losses.append(val_loss)
		cv_iteration.append(best_nround)
        
	print('n validation fold results: {}'.format(cv_losses))

	print('Average iterations: {}'.format(int(np.mean(cv_iteration))))
	print("Mean Cross Validation RMSE: {}\n".format(np.mean(cv_losses)))

	return (oof_train,cv_losses) if get_oof else cv_losses

