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
from sklearn.datasets import make_regression
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

def mse(y_pred, y):
    diffs = np.subtract(y_pred, y)
    m = np.sum(np.square(diffs))
    return m/len(diffs)

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