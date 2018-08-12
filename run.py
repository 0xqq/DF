#导入相关包
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# 读取个人信息
train_agg = pd.read_csv('data/train_agg.csv',sep='\t')
test_agg = pd.read_csv('data/test_agg.csv',sep='\t')
agg = pd.concat([train_agg,test_agg],copy=False)

# 日志信息
train_log = pd.read_csv('data/train_log.csv',sep='\t')
test_log = pd.read_csv('data/test_log.csv',sep='\t')
df_log = pd.concat([train_log,test_log],copy=False)


# 用户唯一标识
train_flg = pd.read_csv('data/train_flg.csv',sep='\t')
test_flg = pd.read_csv('data/submit_sample.csv',sep='\t')
test_flg['FLAG'] = -1
del(test_flg['RST'])
flg = pd.concat([train_flg,test_flg],copy=False)

data = pd.merge(agg,flg,on=['USRID'],how='left',copy=False)
data = pd.merge(agg,flg,on=['USRID'],how='left',copy=False)

# 这里对agg表稍微做了处理，因为统计到V2,V4,V5就2种类型，可能之一为年龄或者等价为年龄的特征，然后V26的总类型数目，我认为是年龄，
# 然后将其按照下面的bins切分，并和V2,V4,V5组合起来，切分成中年男、少年男、老年男等等。当然下面的V22也有可能是类似年龄的特征，
# 也做了相同处理。


bins = [-1,1,2.3,5,11,100]
a = pd.cut(data['V26'],bins)
label = LabelEncoder()
data['V26'] = label.fit_transform(a)

#组合特征
data['V4_V26'] = data['V4'] + data['V26'] 
label = LabelEncoder()
data['V4_V26'] = label.fit_transform(data['V4_V26'])

data['V2_V26'] = data['V2'] + data['V26'] 
label = LabelEncoder()
data['V2_V26'] = label.fit_transform(data['V2_V26'])

data['V5_V26'] = data['V5'] + data['V26'] 
label = LabelEncoder()
data['V5_V26'] = label.fit_transform(data['V5_V26'])

bins = [-1,0,1,2.3,4,10,100]
a = pd.cut(data['V22'],bins)
label = LabelEncoder()
data['V22'] = label.fit_transform(a)
# 先统一把时间换算为秒，便于后面的计算

import time
log = pd.concat([train_log,test_log],copy=False)
a = log['OCC_TIM'].apply(lambda x:time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))

date_1_time = '2018-04-01 00:00:00'
date_8_time = '2018-04-08 00:00:00'
b_1 = time.mktime(time.strptime(date_1_time, "%Y-%m-%d %H:%M:%S"))
b_8 = time.mktime(time.strptime(date_8_time, "%Y-%m-%d %H:%M:%S"))
log = pd.concat([train_log,test_log],copy=False)

# 这个部分计算当前点击APP的时间距离4月1号和4月8号的时间差的统计特征，如均值、方差、最大、最小、中位数、偏度、峰度。这些统计特征
# 可以充分挖掘到不同用户的行为差异。


log['OCC_TIM_1'] = (a - b_1).apply(np.abs)
m = log.groupby(['USRID'],as_index=False)['OCC_TIM_1'].agg({
    'OCC_TIM_1_mean':np.mean,
    'OCC_TIM_1_std':np.std,
    'OCC_TIM_1_min':np.min,
    'OCC_TIM_1_max':np.max,
    'OCC_TIM_1_median':np.median,
    'OCC_TIM_1_skew':skew,
    'OCC_TIM_1_kurtosis':kurtosis
})
data = pd.merge(data,m,on=['USRID'],how='left',copy=False)

log['OCC_TIM_8'] = (a - b_8).apply(np.abs)
n = log.groupby(['USRID'],as_index=False)['OCC_TIM_8'].agg({
    'OCC_TIM_8_mean':np.mean,
    'OCC_TIM_8_std':np.std,
    'OCC_TIM_8_min':np.min,
    'OCC_TIM_8_max':np.max,
    'OCC_TIM_8_median':np.median,
    'OCC_TIM_8_skew':skew,
    'OCC_TIM_8_kurtosis':kurtosis
})
data = pd.merge(data,n,on=['USRID'],how='left',copy=False)
log = pd.concat([train_log,test_log],copy=False)

# 这个部分利用点击的APP时间进行排序。然后计算用户下一次点击APP的时间差统计特征

log['OCC_TIM'] = a
log = log.sort_values(['USRID','OCC_TIM'])
log['next_time'] = log.groupby(['USRID'])['OCC_TIM'].diff(-1).apply(np.abs)

p = log.groupby(['USRID'],as_index=False)['next_time'].agg({
    'next_time_mean':np.mean,
    'next_time_std':np.std,
    'next_time_min':np.min,
    'next_time_max':np.max,
    'next_time_median':np.median,
    'next_time_skew':skew,
    'next_time_kurtosis':kurtosis
})

data = pd.merge(data,p,on=['USRID'],how='left',copy=False)
data = data.fillna(0)
new = pd.merge(df_log, data[['USRID','FLAG']], on='USRID')

#对于点击模块问题，虽然为数字编码，但是题目意思是说点击模块的3个级别，是存在大小关系的。所以我进行了类似于上述时间的统计特征的计算。
#不过在计算之前，是先将3个模块分开，分别计算每个模块的统计特征。

new['EVT_LBL_1'] = new['EVT_LBL'].apply(lambda x:int(x.split('-')[0]))
new['EVT_LBL_2'] = new['EVT_LBL'].apply(lambda x:int(x.split('-')[1]))
new['EVT_LBL_3'] = new['EVT_LBL'].apply(lambda x:int(x.split('-')[2]))

new_1 = new.groupby(['USRID'],as_index=False)['EVT_LBL_1'].agg({
    'EVT_LBL_1_mean':np.mean,
    'EVT_LBL_1_std':np.std,
    'EVT_LBL_1min':np.min,
    'EVT_LBL_1_max':np.max,
    'EVT_LBL_1_median':np.median,
    'EVT_LBL_1_skew':skew,
    'EVT_LBL_1_kurtosis':kurtosis
})
data = pd.merge(data,new_1,on=['USRID'],how='left',copy=False)

new_2 = new.groupby(['USRID'],as_index=False)['EVT_LBL_2'].agg({
    'EVT_LBL_2_mean':np.mean,
    'EVT_LBL_2_std':np.std,
    'EVT_LBL_2min':np.min,
    'EVT_LBL_2_max':np.max,
    'EVT_LBL_2_median':np.median,
    'EVT_LBL_2_skew':skew,
    'EVT_LBL_2_kurtosis':kurtosis
})
data = pd.merge(data,new_2,on=['USRID'],how='left',copy=False)

new_3 = new.groupby(['USRID'],as_index=False)['EVT_LBL_3'].agg({
    'EVT_LBL_3_mean':np.mean,
    'EVT_LBL_3_std':np.std,
    'EVT_LBL_3min':np.min,
    'EVT_LBL_3_max':np.max,
    'EVT_LBL_3_median':np.median,
    'EVT_LBL_3_skew':skew,
    'EVT_LBL_3_kurtosis':kurtosis
})
data = pd.merge(data,new_3,on=['USRID'],how='left',copy=False)

data = data.fillna(0)
new = pd.merge(df_log, data[['USRID','FLAG']], on='USRID')

#这个模块统计了每个有log的用户分别点击了多少种模块

new['EVT_LBL_1'] = new['EVT_LBL'].apply(lambda x:int(x.split('-')[0]))
new['EVT_LBL_2'] = new['EVT_LBL'].apply(lambda x:int(x.split('-')[1]))
new['EVT_LBL_3'] = new['EVT_LBL'].apply(lambda x:int(x.split('-')[2]))

a1 = new.groupby('USRID', as_index=False)['EVT_LBL_1'].count()
data = pd.merge(data, a1, on='USRID', how='left')

a2 = new.groupby('USRID', as_index=False)['EVT_LBL_2'].count()
data = pd.merge(data, a2, on='USRID', how='left')

a3 = new.groupby('USRID', as_index=False)['EVT_LBL_3'].count()
data = pd.merge(data, a3, on='USRID', how='left')

data = data.fillna(0)
#加入每位顾客点击APP次数的特征
c= pd.DataFrame()
c['USRID'] = df_log.USRID.value_counts().index
c['click_total'] = df_log.USRID.value_counts().values
data = pd.merge(data,c,on=['USRID'],how='left',copy=False)
data.loc[data['click_total'].isnull(), 'click_total'] = 0

#加入顾客是如何点击的特征，两种方式：0  2, 缺失值的顾客将其记为-1
b = df_log.groupby(['USRID'], as_index=False)['TCH_TYP'].sum()
data = pd.merge(data, b, on = 'USRID', how='left')
data.loc[data.TCH_TYP > 0, 'TCH_TYP'] = 2.0
data['TCH_TYP'].loc[data['TCH_TYP'].isnull()] = -1
le = LabelEncoder()
data['TCH_TYP'] = le.fit_transform(data['TCH_TYP'])
#这个特征是顾客在3月双休日点击APP的总次数
new = pd.merge(df_log, data[['USRID','FLAG']], on='USRID')

new['date'] = new.OCC_TIM.apply(lambda x:x.split()[0])
new['date'] = new['date'].apply(lambda x:x.split('-')[2])
new['date_week'] = new.date.apply(lambda x:1 if x in ['03','04','17', '18','24','25','31'] else 0)
a = new.groupby('USRID', as_index=False)['date_week'].sum()
data = pd.merge(data, a, on='USRID',how='left')

data = data.fillna(0)
#上面漏了一个agg的处理，也就是将V12进行7等分等频率切割，V2、V4、V5分别两两组合以下作为新的特征

data['V12'] = pd.qcut(data['V12'], 7)
label = LabelEncoder()
data['V12'] = label.fit_transform(data['V12'])

#组合特征
data['V4_V5'] = data['V4'] + data['V5'] 
data['V2_V4'] = data['V2'] + data['V4'] 
data['V2_V5'] = data['V2'] + data['V5']
#这些特征是为了计算用户点击APP的时间段，比如晚上22点到凌晨1点这个时间段用户点击APP的总次数。
#分别考虑了时针和秒针。

new['date_h_1'] = new.OCC_TIM.apply(lambda x:x.split()[1])
new['date_h_1'] = new['date_h_1'].apply(lambda x:x.split(':')[0])
new['date_h_1'] = new['date_h_1'].apply(lambda x:1 if int(x) < 1 else 0)

new['date_h_2'] = new.OCC_TIM.apply(lambda x:x.split()[1])
new['date_h_2'] = new['date_h_2'].apply(lambda x:x.split(':')[0])
new['date_h_2'] = new['date_h_2'].apply(lambda x:1 if int(x) > 22 else 0)

new['date_m_1'] = new.OCC_TIM.apply(lambda x:x.split()[1])
new['date_m_1'] = new['date_m_1'].apply(lambda x:x.split(':')[1])
new['date_m_1'] = new['date_m_1'].apply(lambda x:1 if int(x) < 4 else 0)

new['date_m_2'] = new.OCC_TIM.apply(lambda x:x.split()[1])
new['date_m_2'] = new['date_m_2'].apply(lambda x:x.split(':')[1])
new['date_m_2'] = new['date_m_2'].apply(lambda x:1 if int(x) > 57 else 0)


new['date_add_h'] =new['date_h_2']+new['date_h_1']
a1 = new.groupby('USRID', as_index=False)['date_add_h'].sum()
data = pd.merge(data, a1, on='USRID',how='left')

new['date_add_m'] =new['date_m_2'] + new['date_m_1']
a1 = new.groupby('USRID', as_index=False)['date_add_m'].sum()
data = pd.merge(data, a1, on='USRID',how='left')

data = data.fillna(0)
# 上面就是模型的总特征，下面提取测试集和训练集合，用lightgbm进行预测，由于交叉验证的折数不好确定，我将模型分为5折和10折，然后对两个结果
# 取平均作为最后预测的结果。

train = data.loc[(data['FLAG']!=-1)]
test = data.loc[(data['FLAG']==-1)]
print('train',train.shape)
print('test',test.shape)

# 构造数据
# 提取userid和单独把标签赋值一个变量
train_userid = train.pop('USRID')
y = train.pop('FLAG')
col = train.columns
X = train[col].values

test_userid = test.pop('USRID')
test_y = test.pop('FLAG')
test_X = test[col].values

#归一化
std_scaler = StandardScaler()
X = std_scaler.fit_transform(X)
test_X = std_scaler.transform(test_X)

N = 10
skf = StratifiedKFold(n_splits=N,shuffle=False,random_state=42)

xx_cv = []
xx_pre = []

import operator

for k,(train_in,test_in) in enumerate(skf.split(X,y)):
    X_train,X_test,y_train,y_test = X[train_in],X[test_in],y[train_in],y[test_in]

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # specify your configurations as a dict
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'num_leaves': 32,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }

    print('Start training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=40000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=50,
                   verbose_eval=1500)

    print('Start predicting...')
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    xx_cv.append(roc_auc_score(y_test,np.array(y_pred)))
    xx_pre.append(gbm.predict(test_X, num_iteration=gbm.best_iteration))

xx_pre_yu_10 = np.mean(xx_pre, axis=0)
xx_cv_10 = np.mean(xx_cv)
# 提取测试集和训练集合

train = data.loc[(data['FLAG']!=-1)]
test = data.loc[(data['FLAG']==-1)]
print('train',train.shape)
print('test',test.shape)

# 构造数据
# 提取userid和单独把标签赋值一个变量
train_userid = train.pop('USRID')
y = train.pop('FLAG')
col = train.columns
X = train[col].values

test_userid = test.pop('USRID')
test_y = test.pop('FLAG')
test_X = test[col].values

#还是需要归一化的
std_scaler = StandardScaler()
X = std_scaler.fit_transform(X)
test_X = std_scaler.transform(test_X)

N = 5
skf = StratifiedKFold(n_splits=N,shuffle=False,random_state=42)

xx_cv = []
xx_pre = []

import operator

for k,(train_in,test_in) in enumerate(skf.split(X,y)):
    X_train,X_test,y_train,y_test = X[train_in],X[test_in],y[train_in],y[test_in]

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # specify your configurations as a dict
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'num_leaves': 32,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }

    print('Start training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=40000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=50,
                   verbose_eval=1500)

    print('Start predicting...')
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    xx_cv.append(roc_auc_score(y_test,np.array(y_pred)))
    xx_pre.append(gbm.predict(test_X, num_iteration=gbm.best_iteration))

xx_pre_yu = np.mean(xx_pre, axis=0)
xx_cv = np.mean(xx_cv)
xx_pre_yu = (xx_pre_yu + xx_pre_yu_10) / 2
result = xx_pre_yu
res = pd.DataFrame()
res['USRID'] = list(test_userid.values)
res['RST'] = list(result)

time_date = time.strftime('%Y-%m-%d',time.localtime(time.time()))
res.to_csv('%s_%s.csv'%(str(time_date),str((xx_cv + xx_cv_10)/2).split('.')[1]),index=False,sep='\t')
