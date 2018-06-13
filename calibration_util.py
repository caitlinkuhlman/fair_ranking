import pickle
import numpy as np
import pandas as pd
import random
import math
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import datasets
#from sklearn.naive_bayes import GaussianNB
#from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,f1_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.cross_validation import cross_val_predict
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cross_validation import *
from sklearn.model_selection import ShuffleSplit
#from sklearn.ensemble import AdaBoostRegressor
# from sklearn.ensemble import FairBoostRegressor
#from sklearn.tree import DecisionTreeRegressor
# from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.linear_model import SGDClassifier

#Error metrics

#summary of deviation measures - relates to precision/accuracy?
# https://en.wikipedia.org/wiki/Deviation_(statistics)#Unsigned_or_absolute_deviation
#https://en.wikipedia.org/wiki/Mean_signed_deviation
#signed absolute deviation ?
# https://en.wikipedia.org/wiki/Average_absolute_deviation
#average absolute deviation

def get_ae(vals):
    return np.sum([math.fabs(x[0]-x[1]) for x in vals])

def get_mae(vals):
    m = np.sum([math.fabs(x[0]-x[1]) for x in vals])
    return m/len(vals)


def get_se(vals):
    return np.sum([math.pow(x[0]-x[1], 2) for x in vals])
    
def get_mse(vals):
    m = np.sum([math.pow(x[0]-x[1], 2) for x in vals])
    return m/len(vals)

def mse(y_pred, y):
    diffs = np.subtract(y_pred, y)
    m = np.sum(np.square(diffs))
    return m/len(diffs)

#overestimate
def get_oe(vals):
    return np.sum([max(0,x[1]-x[0]) for x in vals])

def get_moe(vals):
    m = np.sum([max(0,x[1]-x[0]) for x in vals])
    return m/len(vals)

#underestimate
def get_ue(vals):
    return np.sum([max(0,x[0]-x[1]) for x in vals])

def get_mue(vals):
    m = np.sum([max(0,x[0]-x[1]) for x in vals])
    return m/len(vals)

def get_spear(vals):
    a=vals[:,0]
    b=vals[:,1]
    #return stats.spearmanr(a,b)[0]
    return np.cov(a,b)[0][1] / (np.std(a)*np.std(b))
    
def get_kendall(vals):
    a=[x if x != 0 else 0.00001 for x in vals[:,0] ]
    b=[x if x != 0 else 0.00001 for x in vals[:,1] ]
    return stats.kendalltau(a,b, nan_policy='raise')[0]

def get_kl(vals):
    a=[x if x != 0 else 0.00001 for x in vals[:,0] ]
    b=[x if x != 0 else 0.00001 for x in vals[:,1] ]
    return stats.entropy(a, qk=b)

error_functs = [get_ae, get_mae, get_se, get_mse, get_oe, get_moe, get_ue, get_mue]
##### BIN ERRORS: ###########

def get_bin_width(data, n):
    return (data.max()-data.min())/(n+1)    

def get_error_binned_eq_depth_by_group(points, nbins, error=get_mse):
    mse = []
    points = points.sort_values("y_pred")
    kf = KFold(n_splits=nbins, shuffle=False, random_state=1)
    for rest, bin in kf.split(points):
        vals = points.iloc[bin][["y_pred","y"]]
        mse.append(error(np.array(vals)))
    return mse

def plot_binned_error(df, error=get_mse):
    indices =np.arange(df.shape[0])
    #Calculate optimal width
    width = 0.3

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(indices-width,df[df.columns[0]],width,color='b',label='-Ymin')
    ax.bar(indices,df[df.columns[1]],width,color='r',label='Ymax')
    ax.set_xlim(left=-1,right=len(df))
    ax.set_ylim([0,1])
    ax.set_xlabel('Bin')
    ax.set_ylabel(error.__name__)
    plt.show()
    
def recomputeBins_no_groups(y_predict, nbins):
        #g0 has indexes of objects in X
        # get indexes of sorted predictions for group
        sorted = np.argsort(y_predict)
        binSize=int(np.ceil(float(len(y_predict))/nbins))
        bins=[]
        b=[]
        i=0
        j=binSize-1
        for n in range(nbins):
            k=int(min(j,len(sorted)-1))
            bins.append(sorted0[i:k])
            i+=binSize
            j+=binSize
           
        return bins

def get_calibration_error(y_pred, y, nbins):    
    error_vect=np.abs(y_pred - y)
    e0=[np.sum([error_vect[i] for i in b]) for b in bins0]
    bin_error = np.subtract(e0, e1)
    return np.mean(np.abs(bin_error))

def recomputeBins(y_predict, g0, g1, nbins):
        sorted0 = np.argsort([y_predict[x] for x in g0])
        binSize=int(np.ceil(float(len(g0))/nbins))
        bins0=[]
        b=[]
        i=0
        j=binSize-1
        for n in range(nbins):
            k=int(min(j,len(sorted0)-1))
            bins0.append([g0[x] for x in sorted0[i:k]])
            i+=binSize
            j+=binSize
        sorted1 = np.argsort([y_predict[x] for x in g1])
        binSize=int(np.ceil(float(len(g1))/nbins))
        bins1=[]
        b=[]
        i=0
        j=binSize-1
        for n in range(nbins):
            k=int(min(j,len(sorted0)-1))
            bins1.append([g1[x] for x in sorted1[i:k]])
            i+=binSize
            j+=binSize
        return bins0,bins1
        
def get_fair_error(y_pred, y, g0, g1, nbins):    
    bins0,bins1  = recomputeBins(y_pred, g0, g1, nbins)
    error_vect=np.abs(y_pred - y)
    e0=[np.sum([error_vect[i] for i in b]) for b in bins0]
    e1=[np.sum([error_vect[i] for i in b]) for b in bins1]
    bin_error = np.subtract(e0, e1)
    return np.mean(np.abs(bin_error))
    
def get_errors(y_hat, y, g, nbins):
    results = pd.DataFrame()
    results['pred'] = y_hat
    results['y'] = y
    results['g'] = g
    results = results.sort_values('pred')
    results0 = results[results['g']==0.]
    results1 = results[results['g']==1.]
    errors = pd.DataFrame()
    errors['mse0'] = get_error_binned_eq_depth_by_group(results0, 
                                                        nbins, error=get_mse)
    errors['mse1'] = get_error_binned_eq_depth_by_group(results1, 
                                                        nbins, error=get_mse)
    errors['bin_mse_diff']=errors['mse0']-errors['mse1']
    errors['ae0'] = get_error_binned_eq_depth_by_group(results0, 
                                                      nbins, error=get_ae)
    errors['ae1'] = get_error_binned_eq_depth_by_group(results1, 
                                                        nbins, error=get_ae)
    errors['bin_ae_diff']=errors['ae0']-errors['ae1']
    errors['oe0'] = get_error_binned_eq_depth_by_group(results0, 
                                                      nbins, error=get_oe)
    errors['oe1'] = get_error_binned_eq_depth_by_group(results1, 
                                                        nbins, error=get_oe)
    errors['bin_oe_diff']=errors['oe0']-errors['oe1']
    errors['ue0'] = get_error_binned_eq_depth_by_group(results0, 
                                                      nbins, error=get_ue)
    errors['ue1'] = get_error_binned_eq_depth_by_group(results1, 
                                                        nbins, error=get_ue)
    errors['bin_ue_diff']=errors['ue0']-errors['ue1']
    errors['spear0'] = get_error_binned_eq_depth_by_group(results0, 
                                                      nbins, error=get_spear)
    errors['spear1'] = get_error_binned_eq_depth_by_group(results1, 
                                                        nbins, error=get_spear)
    errors['bin_spear_diff']=errors['spear0']-errors['spear1']
    return errors
    
def get_calibrated_predictions(train,test):
    
    ir = IsotonicRegression( out_of_bounds = 'clip' )
    ir.fit(train['y_pred'], train['y'])
    
    return ir.transform(test['y_pred'])  
    
def get_groupwise_calibrated_predictions(train, test):
    
    y_pred0 = train[train['g']==0.]['y_pred']
    y_pred1 = train[train['g']==1.]['y_pred']
    y_0 = train[train['g']==0.]['y']
    y_1 = train[train['g']==1.]['y']
    ir0 = IsotonicRegression( out_of_bounds = 'clip' )
    ir0.fit(y_pred0, y_0)
    ir1 = IsotonicRegression( out_of_bounds = 'clip' )
    ir1.fit(y_pred1, y_1)
    
    test0 = test[test['g']==0.]['y_pred']
    test1 = test[test['g']==1.]['y_pred']
    return ir0.transform(test0), ir1.transform(test1)

def scale(data, col):
    data[col] = (data[col]-data[col].min()) / (data[col].max() - data[col].min())
    
# generate regression dataset
def generate(n_points, n_attr, n, r):
    data = pd.DataFrame()
    X1, y1 = make_regression(n_samples=n_points, n_features=n_attr, noise=n, random_state=r)
    g1 = np.zeros(int(len(X1)/2))
    g2 = np.ones(int(len(X1)/2))
    g = np.append(g1, g2)
    d1 = np.insert(X1, 5, values=y1, axis=1)
    data = pd.DataFrame(d1)
    data = data.sample(frac=1)
    data.columns = ['x0','x1','x2','x3','x4','y']
    data['g']=g
    #set y between 0 and 1,sort
    scale(data, 'y')
    data.sort_values('y', inplace=True)
    return data
    
def normalize(df):
    return(df - df.mean()) / df.std()

def split_data_norm(data):
    test = data.sample(frac=0.5, random_state=42)
    test=normalize(test)
    train_all = data.drop(test.index)
    train_all=normalize(train_all)
    tune = train_all.sample(frac=0.2, random_state=42)
    train = train_all.drop(tune.index)
    print("test ", test.shape)
    print("train_all ",train_all.shape)
    print("tune ",tune.shape)
    print("train ",train.shape)
    return np.nan_to_num(test), np.nan_to_num(train_all), np.nan_to_num(tune), np.nan_to_num(train)

def split_data(data):
    test = data.sample(frac=0.5, random_state=42)
    train_all = data.drop(test.index)
    #1,000
    tune = train_all.sample(frac=0.2, random_state=42)
    train = train_all.drop(tune.index)
#     print("test ", test.shape)
#     print("train_all ",train_all.shape)
#     print("tune ",tune.shape)
#     print("train ",train.shape)
    return np.nan_to_num(test), np.nan_to_num(train_all), np.nan_to_num(tune), np.nan_to_num(train)


def under_rank(data, sample, scale, group):
    
    #randomly select some percentage of group to be under ranked
    data2 = data.copy()
    for i in range(len(data)):
        if(data2.iloc[i]['g'] == group):
            if(random.random() <sample):
                data2.iloc[i]['y'] = data2.iloc[i]['y']*(random.random()+scale)

    data2.sort_values('y', inplace=True)
    return data2

def get_bin_errors(y_hat, y, g, nbins):
    results = pd.DataFrame()
    results['y_pred'] = y_hat
    results['y'] = y
    results['g'] = g
    results = results.sort_values('y_pred')
    results0 = results[results['g']==0.]
    results1 = results[results['g']==1.]
    
    errors = pd.DataFrame()
    errors['mse0'] = get_error_binned_eq_depth_by_group(results0, 
                                                        nbins, error=get_mse)
    errors['mse1'] = get_error_binned_eq_depth_by_group(results1, 
                                                        nbins, error=get_mse)
    errors['bin_mse_diff']=(errors['mse0']-errors['mse1']).abs()
    errors['ae0'] = get_error_binned_eq_depth_by_group(results0, 
                                                      nbins, error=get_ae)
    errors['ae1'] = get_error_binned_eq_depth_by_group(results1, 
                                                        nbins, error=get_ae)
    
    errors['bin_ae_diff']=(errors['ae0']-errors['ae1']).abs()
    errors['oe0'] = get_error_binned_eq_depth_by_group(results0, 
                                                      nbins, error=get_oe)
    errors['oe1'] = get_error_binned_eq_depth_by_group(results1, 
                                                        nbins, error=get_oe)
    errors['bin_oe_diff']=(errors['oe0']-errors['oe1']).abs()
    errors['ue0'] = get_error_binned_eq_depth_by_group(results0, 
                                                      nbins, error=get_ue)
    errors['ue1'] = get_error_binned_eq_depth_by_group(results1, 
                                                        nbins, error=get_ue)
    errors['bin_ue_diff']=(errors['ue0']-errors['ue1']).abs()
    
    #need to convert scores to ranks to compute ranking correlation
    ranks= results.copy()
    ranks["y_pred"]=results["y_pred"].rank(method="average")
    ranks["y"]=results["y"].rank(method="average")
    ranks0 = results[ranks['g']==0.]
    ranks1 = results[ranks['g']==1.]
    
    errors['spear0'] = get_error_binned_eq_depth_by_group(ranks0, 
                                                      nbins, error=get_spear)
    errors['spear1'] = get_error_binned_eq_depth_by_group(ranks1, 
                                                        nbins, error=get_spear)
    errors['bin_spear_diff']=(errors['spear0']-errors['spear1']).abs()
    
    
    errors['kendall0'] = get_error_binned_eq_depth_by_group(ranks0, 
                                                      nbins, error=get_kendall)
    errors['kendall1'] = get_error_binned_eq_depth_by_group(ranks1, 
                                                        nbins, error=get_kendall)
    errors['bin_kendall_diff']=(errors['kendall0']-errors['kendall1']).abs()
    
    
    errors['kl0'] = get_error_binned_eq_depth_by_group(ranks0, 
                                                      nbins, error=get_kl)
    errors['kl1'] = get_error_binned_eq_depth_by_group(ranks1, 
                                                        nbins, error=get_kl)
    errors['bin_kl_diff']=(errors['kl0']-errors['kl1']).abs()
    return errors[['bin_mse_diff','bin_ae_diff', 'bin_oe_diff', 'bin_ue_diff', 
                   'bin_spear_diff','bin_kendall_diff', 'bin_kl_diff']]

#use 25% of the data for testing, and rest for calibration
def train_test_short(df, folds, nbins, r):
    cv_errors =pd.DataFrame()
    train = df.sample(frac=0.25)
    test = df.drop(train.index)

    y_pred_baseline = test['y_pred']

    y_pred_cal = get_calibrated_predictions(train, test)
    
    y_pred0, y_pred1 = get_groupwise_calibrated_predictions(train, test)
    y_pred_all = np.append(y_pred0, y_pred1)

    y_test0 = test[test['g']==0.]['y']
    y_test1 = test[test['g']==1.]['y']
    y_test_all = np.append(y_test0, y_test1)

    g_test0 = test[test['g']==0.]['g']
    g_test1 = test[test['g']==1.]['g']
    g_test_all = np.append(g_test0, g_test1)

    y_test = np.array(test['y'])
    g_test = np.array(test['g'])

    b_errors = get_bin_errors(y_pred_baseline, y_test, g_test, nbins).mean()
    cv_errors["baseline"] = b_errors
    c_errors = get_bin_errors(y_pred_cal, y_test, g_test, nbins).mean()
    cv_errors["calibrate"] = c_errors
    cg_errors = get_bin_errors(y_pred_all, y_test_all, g_test_all, nbins).mean()
    cv_errors["cal_groups"] = cg_errors
    cv_errors.index = ['Mean Squared Error', 'Absolute Error', 'Overestimation','Underestimation', 'Spearman Rho', 'Kendall Tau', 'KL divergence']

    return cv_errors

def stabalize_cv(df, folds, nbins):
    df1 = cv_short(df, folds, nbins)
    df2 = cv_short(df, folds, nbins)
    df3 = cv_short(df, folds, nbins)
    return (df1+df2+df3)/3

#use 25% of the data for testing, and rest for calibration
def cv_short(df, folds, nbins):
    cv_errors =pd.DataFrame()
    cv_errors["baseline"] = np.zeros(7)
    cv_errors["calibrate"] = np.zeros(7)
    cv_errors["cal_groups"] = np.zeros(7)
    kf = KFold(n_splits=folds, shuffle=True)
    
    for train_index, test_index in kf.split(df):
        train = df.iloc[train_index]
        test = df.drop(train.index)

        y_pred_baseline = test['y_pred']

        y_pred_cal = get_calibrated_predictions(train, test)

        y_pred0, y_pred1 = get_groupwise_calibrated_predictions(train, test)
        y_pred_all = np.append(y_pred0, y_pred1)

        y_test0 = test[test['g']==0.]['y']
        y_test1 = test[test['g']==1.]['y']
        y_test_all = np.append(y_test0, y_test1)

        g_test0 = test[test['g']==0.]['g']
        g_test1 = test[test['g']==1.]['g']
        g_test_all = np.append(g_test0, g_test1)

        y_test = np.array(test['y'])
        g_test = np.array(test['g'])

        b_errors = np.array(get_bin_errors(y_pred_baseline, y_test, g_test, nbins).mean())
        cv_errors["baseline"] = cv_errors["baseline"] + b_errors
        c_errors = np.array(get_bin_errors(y_pred_cal, y_test, g_test, nbins).mean())
        cv_errors["calibrate"] = cv_errors["calibrate"] + c_errors
        cg_errors = np.array(get_bin_errors(y_pred_all, y_test_all, g_test_all, nbins).mean())
        cv_errors["cal_groups"] = cv_errors["cal_groups"] +cg_errors
        cv_errors.index = ['Mean Squared Error', 'Absolute Error', 'Overestimation','Underestimation', 'Spearman Rho', 'Kendall Tau', 'KL divergence']

    cv_errors=cv_errors / folds
    cv_errors.index = ['Mean Squared Error', 'Absolute Error', 'Overestimation','Underestimation', 'Spearman Rho', 'Kendall Tau', 'KL divergence']

    return cv_errors


def run(data_g,data_y,data_X):
    test_g, train_all_g, tune_g, train_g = split_data(data_g)
    test_y, train_all_y, tune_y, train_y = split_data_norm(data_y)
    test_X, train_all_X, tune_X, train_X = split_data_norm(data_X)
    
    lr = LinearRegression()
    #linear regression
    lr.fit(train_all_X, train_all_y)
    lr_y_pred = lr.predict(test_X)
    lr_err = mse(lr_y_pred, test_y)
    print("linear regression mse: ", lr_err)
    
    df_lr=pd.DataFrame()
    df_lr['y']=test_y
    df_lr['y_pred']=lr_y_pred
    df_lr['g']=test_g
    
    
    #LASSO
    errs=[]
    depths = [0.1, 0.3,0.5,0.7,1]
    for depth in depths:
        kf = KFold(n_splits=5, shuffle=True, random_state=99)
        mses =[]
        for train_index, test_index in kf.split(tune_X):
            la = Lasso(alpha=depth, fit_intercept=True, max_iter=1000, tol=0.0001)
            la.fit(tune_X[train_index], tune_y[train_index])
            la_y_pred = la.predict(tune_X[test_index])
            mses.append(mse(la_y_pred, tune_y[test_index]))
        errs.append(np.sum(mses))
    best = np.argmin(errs)
    print("best alpha: ", depths[best])
    
    la = Lasso(alpha=depths[best], fit_intercept=True, max_iter=1000, tol=0.0001)
    la.fit(train_X[train_index], train_y[train_index])

    la_y_pred = la.predict(test_X)
    la_err = mse(la_y_pred, test_y)
    print("lasso mse: ", la_err)
    
    df_la=pd.DataFrame()
    df_la['y']=test_y
    df_la['y_pred']=la_y_pred
    df_la['g']=test_g
    
    #DECISON TREE
    errs=[]
    depths = [1,2,3,5,7,9]
    for depth in depths:
        kf = KFold(n_splits=5, shuffle=True, random_state=99)
        mses =[]
    
        for train_index, test_index in kf.split(tune_X):
            dt = DecisionTreeRegressor(max_depth=depth)
            dt.fit(tune_X[train_index], tune_y[train_index])
            y_pred = dt.predict(tune_X[test_index])
            y = tune_y[test_index]
            mses.append(mse(y_pred, tune_y[test_index]))
        
        errs.append(np.sum(mses))
    best = np.argmin(errs)
    print("best depth of tree: ", depths[best])
    
    dt = DecisionTreeRegressor(max_depth=depths[best])
    dt.fit(train_X[train_index], train_y[train_index])

    dt_y_pred = dt.predict(test_X)
    dt_err = mse(dt_y_pred, test_y)
    print("decision tree mse: ", dt_err)
    
    df_dt=pd.DataFrame()
    df_dt['y']=test_y
    df_dt['y_pred']=dt_y_pred
    df_dt['g']=test_g
    
    
    #RANDOM FOREST
    errs=[]
    depths = [1,2,3,5,7,9]
    for depth in depths:
        mses =[]
        for train_index, test_index in kf.split(tune_X):
            rf = RandomForestRegressor(max_depth=depth)
            rf.fit(tune_X[train_index], tune_y[train_index])
            y_pred = rf.predict(tune_X[test_index])
            y = tune_y[test_index]
            mses.append(mse(y_pred, tune_y[test_index]))
        
        errs.append(np.sum(mses)/len(mses))
    best = np.argmin(errs)
    print("best depth of tree: ", depths[best])
    
    rf = RandomForestRegressor(max_depth=depths[best])
    rf.fit(train_X[train_index], train_y[train_index])

    rf_y_pred = rf.predict(test_X)
    rf_err = mse(rf_y_pred, test_y)
    print("random forest mse: ", rf_err)

    df_rf=pd.DataFrame()
    df_rf['y']=test_y
    df_rf['y_pred']=rf_y_pred
    df_rf['g']=test_g
    
    #LINEAR SVM
    errs=[]
    depths = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    for depth in depths:
        mses =[]
        for train_index, test_index in kf.split(tune_X):
            svr = LinearSVR(epsilon=0.0, tol=0.0001, loss= 'squared_epsilon_insensitive', C=depth, dual=False, max_iter=1000)
            svr.fit(tune_X[train_index], tune_y[train_index])
            y_pred = svr.predict(tune_X[test_index])
            y = tune_y[test_index]
            mses.append(mse(y_pred, tune_y[test_index]))
            
        errs.append(np.sum(mses)/len(mses))
    best = np.argmin(errs)
    print("Best C value: ", depths[best])
    
    svr = LinearSVR(epsilon=0.0, tol=0.0001, loss='squared_epsilon_insensitive', C=depths[best], dual=False, max_iter=1000) 
    svr.fit(train_X[train_index], train_y[train_index])

    svr_y_pred = svr.predict(test_X)
    svr_err = mse(svr_y_pred, test_y)
    print("linear svm: ", svr_err)
    
    df_svr=pd.DataFrame()
    df_svr['y']=test_y
    df_svr['y_pred']=svr_y_pred
    df_svr['g']=test_g
    
    return df_la, df_dt, df_rf, df_svr

#df = ['y_pred','y','g']
def sliding_mse(df, window, step):
    df.sort_values('y_pred', ascending=False, inplace=True)
    err0=[]
    err1=[]
    start=0
    end=window
    while end<len(df):
        vals = df.iloc[range(start,end)]
        g0 = np.array(vals[vals['g']==0.][['y','y_pred']])
        g1 = np.array(vals[vals['g']==1.][['y','y_pred']])
        err0.append(get_mse(g0))
        err1.append(get_mse(g1))
        start+=step
        end+=step
    #get end of rank is needed
    if(start > len(df)-window):
        vals = df.iloc[range(len(df)-window,len(df))]
        g0 = np.array(vals[vals['g']==0.][['y','y_pred']])
        g1 = np.array(vals[vals['g']==1.][['y','y_pred']])
        err0.append(get_mse(g0))
        err1.append(get_mse(g1))
    return err0, err1

#df = ['y_pred','y','g']
def sliding_oe(df, window, step):
    df.sort_values('y_pred', ascending=False, inplace=True)
    err0=[]
    err1=[]
    start=0
    end=window
    while end<len(df):
        vals = df.iloc[range(start,end)]
        g0 = np.array(vals[vals['g']==0.][['y','y_pred']])
        g1 = np.array(vals[vals['g']==1.][['y','y_pred']])
        err0.append(get_oe(g0))
        err1.append(get_oe(g1))
        start+=step
        end+=step
    #get end of rank is needed
    if(start > len(df)-window):
        vals = df.iloc[range(len(df)-window,len(df))]
        g0 = np.array(vals[vals['g']==0.][['y','y_pred']])
        g1 = np.array(vals[vals['g']==1.][['y','y_pred']])
        err0.append(get_oe(g0))
        err1.append(get_oe(g1))
    return err0, err1

#df = ['y_pred','y','g']
def sliding_ue(df, window, step):
    df.sort_values('y_pred', ascending=False, inplace=True)
    err0=[]
    err1=[]
    start=0
    end=window
    while end<len(df):
        vals = df.iloc[range(start,end)]
        g0 = np.array(vals[vals['g']==0.][['y','y_pred']])
        g1 = np.array(vals[vals['g']==1.][['y','y_pred']])
        err0.append(get_ue(g0))
        err1.append(get_ue(g1))
        start+=step
        end+=step
    #get end of rank is needed
    if(start > len(df)-window):
        vals = df.iloc[range(len(df)-window,len(df))]
        g0 = np.array(vals[vals['g']==0.][['y','y_pred']])
        g1 = np.array(vals[vals['g']==1.][['y','y_pred']])
        err0.append(get_ue(g0))
        err1.append(get_ue(g1))
    return err0, err1


#df = ['y_pred','y','g']
def sliding_prob(df, window, step):
    df.sort_values('y_pred', ascending=False, inplace=True)
    prob0=[]
    prob1=[]
    start=0
    end=window
    g0=(df['g']==0.).sum()
    g1=(df['g']==1.).sum()
    while end<len(df):
        vals = df.iloc[range(start,end)]
        n0 = (vals['g']==0.).sum()
        n1 = (vals['g']==1.).sum()
        prob0.append(max(n0/(g0/window), 0.0001))
        prob1.append(max(n1/(g1/window), 0.0001))
        start+=step
        end+=step
    #get end of rank is needed
    if(start > len(df)-window):
        vals = df.iloc[range(len(df)-window,len(df))]
        n0 = (vals['g']==0.).sum()
        n1 = (vals['g']==1.).sum()
        prob0.append(max(n0/(g0/window), 0.0001))
        prob1.append(max(n1/(g1/window), 0.0001))
    return prob0, prob1

def get_kl_err(df, err, window=100, step=10):
    err0,err1 = err(df, window, step)
    return stats.entropy(err0, qk=err1)
        
def KL_eval(dfs, err):
    errs =[]
    for df in dfs:
        errs.append(get_kl_err(df, err))
    return errs 

def get_spear_err(df, err, window=100, step=10):
    err0,err1 = err(df, window, step)
    #return stats.spearmanr(a,b)[0]
    return np.cov(err0,err1)[0][1] / (np.std(err0)*np.std(err1))
    
def get_kendall_err(df, err, window=100, step=10):
    err0,err1 = err(df, window, step)
    return stats.kendalltau(err0,err1, nan_policy='raise')[0]

def add_error(data, scale, group):
    data2 = data.copy()
    for i in range(len(data)):
        if(data2.iloc[i]['g'] == group):
            data2.iloc[i]['y'] = data2.iloc[i]['y']*(random.uniform(scale, 1.1))
        else:
            data2.iloc[i]['y'] = data2.iloc[i]['y']*(random.uniform(0.9, 1.1))
    #data2.sort_values('y', inplace=True)   
    return data2


def plot_rank(data, col):
    cmap = plt.cm.rainbow
    plt.rcParams['figure.figsize'] = (20, 4)
    fig, ax = plt.subplots()
    ax.bar(range(len(data)), data[col], 0.5, color=cmap((data['g'])))
    ax.set_xlim([0,len(data)])
    ax.set_title("", fontsize=16)
    plt.show()    


def plot_errs(df, window, step):
    plt.rcParams['figure.figsize'] = (6, 4)
    
    err0,err1 = sliding_prob(df, window, step)
    plt.plot(err0, color='red')
    plt.plot(err1, color='blue')
    plt.title("Sliding Window Statistical Parity")
    plt.show()

    err0,err1 = sliding_mse(df, window, step)
    plt.plot(err0, color='red')
    plt.plot(err1, color='blue')
    plt.title("Sliding Window MSE")
    plt.show()

    err0,err1 = sliding_oe(df, window, step)
    plt.plot(err0, color='red')
    plt.plot(err1, color='blue')
    plt.title("Sliding Window Overestimation")
    plt.show()

    err0,err1 = sliding_ue(df, window, step)
    plt.plot(err0, color='red')
    plt.plot(err1, color='blue')
    plt.title("Sliding Window Underestimation")
    plt.show()