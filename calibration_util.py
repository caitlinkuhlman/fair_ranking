import pickle
import numpy as np
import pandas as pd
import random
import math
from scipy import stats
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
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

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

#overestimate
def get_oe(vals):
    return np.sum([max(0,x[0]-x[1]) for x in vals])

def get_moe(vals):
    m = np.sum([max(0,x[0]-x[1]) for x in vals])
    return m/len(vals)

#underestimate
def get_ue(vals):
    return np.sum([min(0,x[0]-x[1]) for x in vals])

def get_mue(vals):
    m = np.sum([min(0,x[0]-x[1]) for x in vals])
    return m/len(vals)

def get_spear(vals):
    a=[x[0] for x in vals]
    b=[x[1] for x in vals]
    return stats.spearmanr(a,b)[0]

error_functs = [get_ae, get_mae, get_se, get_mse, get_oe, get_moe, get_ue, get_mue]
##### BIN ERRORS: ###########

def get_bin_width(data, n):
    return (data.max()-data.min())/(n+1)    

def get_error_binned_eq_depth_by_group(points, nbins, error=get_mse):
    mse = []
    kf = KFold(n_splits=nbins, shuffle=True, random_state=1)
    for rest, bin in kf.split(points):
        vals = [points.iloc[i] for i in bin]
        mse.append(error(vals))
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
    
def get_baseline_predictions(X_train, y_train, X_test):
    
    lr = linear_model.LinearRegression()
    lr.fit(X_train, y_train)
    
    return lr.predict(X_test) 
    
    
def get_calibrated_predictions(X_train, y_train, X_cal, y_cal, X_test):
    
    lr = linear_model.LinearRegression()
    lr.fit(X_train, y_train)
    
    y_pred = lr.predict(X_cal) 
    ir = IsotonicRegression( out_of_bounds = 'clip' )
    ir.fit(y_pred, y_cal)
    
    return ir.transform(lr.predict(X_test))  
    
def get_groupwise_calibrated_predictions(X_train, y_train, X_cal, y_cal, g_cal, X_test, g_test):
    
    lr = linear_model.LinearRegression()
    lr.fit(X_train, y_train)
    
    y_pred = lr.predict(X_cal) 
    y_pred0 = y_pred[g_cal ==0.]
    y_pred1 = y_pred[g_cal ==1.]
    y_cal0 = y_cal[g_cal ==0.]
    y_cal1 = y_cal[g_cal ==1.]
    X_test0 = X_test[g_test ==0.]
    X_test1 = X_test[g_test ==1.]
    ir0 = IsotonicRegression( out_of_bounds = 'clip' )
    ir0.fit(y_pred0, y_cal0)
    ir1 = IsotonicRegression( out_of_bounds = 'clip' )
    ir1.fit(y_pred1, y_cal1)
    
    y_pred0 = ir0.transform(lr.predict(X_test0))  
    y_pred1 = ir1.transform(lr.predict(X_test1))
    return y_pred0, y_pred1