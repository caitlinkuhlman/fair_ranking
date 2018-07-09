import numpy as np
import pandas as pd
from calibration_util import *

def pairs(n):
    return n*(n-1)/2

#new calibration
def merge_1_cal(h1,h2):
    count = 0
    arr = []
    i=0
    j=0
    while i < len(h1) and j < len(h2):
#         print("compare", h1[i] , h2[j])
        if h1[i][0] > h2[j][0]:
            arr.append(h1[i])
            i+=1
        else:
            # count times discord pair contains group 1
            l=i
            if(h2[j][1] == 1):
                while l < len(h1):
                    count += 1
                    l+=1 
            else:
                while l < len(h1):
                    if h1[l][1] == 1:
                        count += 1
                    l+=1 
            arr.append(h2[j])
            j+=1
            
#     add any remaining elements
    while i < len(h1):
        arr.append(h1[i])
        i+=1
    while j < len(h2):
        arr.append(h2[j])
        j+=1
    return arr, count 

#new calibration
def merge_0_cal(h1,h2):
    count = 0
    arr = []
    i=0
    j=0
    while i < len(h1) and j< len(h2):
#         print("compare", h1[i] , h2[j])
        if h1[i][0] > h2[j][0]:
            arr.append(h1[i])
            i+=1
        else:
            # count times discord pair contains group 0
            l=i
            if(h2[j][1] == 0):
                while l < len(h1):
                    count += 1
#                     print("count", h1[l][0])
                    l+=1 
            else:
                while l < len(h1):
                    if h1[l][1] == 0:
                        count += 1
#                         print("count", h1[l][0])
                    l+=1
            arr.append(h2[j])
            j+=1
            
#     add any remaining elements
    while i < len(h1):
        arr.append(h1[i])
        i+=1
    while j < len(h2):
        arr.append(h2[j])
        j+=1
    return arr, count
    
#equality
def merge_0_eq(h1,h2):
    count = 0
    arr = []
    i=0
    j=0
    while i < len(h1) and j< len(h2):
#         print("compare", h1[i] , h2[j])
        if h1[i][0] > h2[j][0]:
            arr.append(h1[i])
            i+=1
        else:
            # count times group 0 preferred to group 1
            if(h2[j][1] == 1):
                l=i
                while l < len(h1):
                    if h1[l][1] == 0:
#                         print("count", h1[l], h2[j])
                        count += 1
                    l+=1   
            arr.append(h2[j])
            j+=1
            
#     add any remaining elements
    while i < len(h1):
        arr.append(h1[i])
        i+=1
    while j < len(h2):
        arr.append(h2[j])
        j+=1
    return arr, count

#equality
def merge_1_eq(h1,h2):
    count = 0
    arr = []
    i=0
    j=0
    while i < len(h1) and j< len(h2):
#         print("compare", h1[i] , h2[j])
        if h1[i][0] > h2[j][0]:
            arr.append(h1[i])
            i+=1
        else:
            # count times group 1 preferred to group 0
            if(h2[j][1] == 0):
                l=i
                while l < len(h1):
                    if h1[l][1] == 1:
#                         print("count", h1[l], h2[j])
                        count += 1
                    l+=1   
            arr.append(h2[j])
            j+=1
            
#     add any remaining elements
    while i < len(h1):
        arr.append(h1[i])
        i+=1
    while j < len(h2):
        arr.append(h2[j])
        j+=1
    return arr, count 


# data already sorted - just count times group is preferred
def merge_parity_0(h1,h2):
    count0 = 0
    count1 = 0
    i=0
    while i < len(h1): 
        if h1[i][1] == 0:
            count0 +=1
        i+=1
    i=0
    while i < len(h2):
        if h2[i][1] == 1:
            count1 +=1  
        i+=1
    count = count0*count1        
    return np.concatenate([h1,h2]), count

def merge_parity_1(h1,h2):
#     print(h1, h2)
    count0 = 0
    count1 = 0
    i=0
    while i < len(h1): 
        if h1[i][1] == 1:
            count1 +=1
        i+=1
    i=0
    while i < len(h2):
        if h2[i][1] == 0:
            count0 +=1  
        i+=1
#     print(count0, count1)
    count = count0*count1 
    return np.concatenate([h1,h2]), count

def count_inversions(data, s, e, merge_fnc):
    
    if s == e: #base case
        return [data[s]], 0
    else:
        m = s + int((e-s)/2)
#         print(s, m, e)
        h1,c1 = count_inversions(data, s, m, merge_fnc)
        h2,c2 = count_inversions(data, m+1, e, merge_fnc)
        merged, c = merge_fnc(h1,h2)
        return merged, (c1+c2+c)


    
# normalized version
def sliding_kendall_eq(df, window, step):
    df.sort_values('y_pred', ascending=False, inplace=True)
    err0=[]
    err1=[]
    start=0
    end=window
    while end<len(df):
        vals = df.iloc[range(start,end)]
        g = np.array(vals.sort_values('y', ascending=False)[['y_pred', 'g']])
#         normalize by the number of pairs containing one item from each group
        p = pairs(len(g)) - pairs(len(vals[vals['g']==1])) - pairs(len(vals[vals['g']==0]))
        e0 = 0 if p == 0 else count_inversions(g, 0, len(g)-1, merge_0_eq)[1] / p
        err0.append(e0)
        e1 = 0 if p == 0 else count_inversions(g, 0, len(g)-1, merge_1_eq)[1] / p
        err1.append(e1)
        start+=step
        end+=step
    #get end of rank is needed
    if(start > len(df)-window):
        vals = df.iloc[range(len(df)-window,len(df))]
        g = np.array(vals.sort_values('y', ascending=False)[['y_pred', 'g']])
        p = pairs(len(g)) - pairs(len(vals[vals['g']==1])) - pairs(len(vals[vals['g']==0]))
        e0 = 0 if p == 0 else count_inversions(g, 0, len(g)-1, merge_0_eq)[1] / p
        err0.append(e0)
        e1 = 0 if p == 0 else count_inversions(g, 0, len(g)-1, merge_1_eq)[1] / p
        err1.append(e1)
    return err0, err1

# normalized version
def sliding_kendall_cal2(df, window, step):
    df.sort_values('y_pred', ascending=False, inplace=True)
    err0=[]
    err1=[]
    start=0
    end=window
    while end<len(df):
        vals = df.iloc[range(start,end)]
        g = np.array(vals.sort_values('y', ascending=False)[['y_pred', 'g']])
        p0 = pairs(len(g)) - pairs(len(vals[vals['g']==1]))
        p1 = pairs(len(g)) - pairs(len(vals[vals['g']==0]))
        e0 = 0 if p0 == 0 else count_inversions(g, 0, len(g)-1, merge_0_cal)[1] / p0
        err0.append(e0)
        e1 = 0 if p1 == 0 else count_inversions(g, 0, len(g)-1, merge_1_cal)[1] / p1
        err1.append(e1)
        start+=step
        end+=step
    #get end of rank is needed
    if(start > len(df)-window):
        vals = df.iloc[range(len(df)-window,len(df))]
        g = np.array(vals.sort_values('y', ascending=False)[['y_pred', 'g']])
        p0 = pairs(len(g)) - pairs(len(vals[vals['g']==1]))
        p1 = pairs(len(g)) - pairs(len(vals[vals['g']==0]))
        e0 = 0 if p0 == 0 else count_inversions(g, 0, len(g)-1, merge_0_cal)[1] / p0
        err0.append(e0)
        e1 = 0 if p1 == 0 else count_inversions(g, 0, len(g)-1, merge_1_cal)[1] / p1
        err1.append(e1)
    return err0, err1


#normalized version
def sliding_kendall_parity(df, window, step):
    df.sort_values('y_pred', ascending=False, inplace=True)
    err0=[]
    err1=[]
    start=0
    end=window
    while end<len(df):
        vals = df.iloc[range(start,end)]
        g = np.array(vals[['y_pred', 'g']])
        p = pairs(len(g)) - pairs(len(vals[vals['g']==1])) - pairs(len(vals[vals['g']==0]))
        e0 = 0 if p == 0 else count_inversions(g, 0, len(g)-1, merge_parity_0)[1] / p
        err0.append(e0)
        e1 = 0 if p == 0 else count_inversions(g, 0, len(g)-1, merge_parity_1)[1] / p
        err1.append(e1)
        start+=step
        end+=step
    #get end of rank is needed
    if(start > len(df)-window):
        vals = df.iloc[range(len(df)-window,len(df))]
        g = np.array(vals[['y_pred', 'g']])
        p = pairs(len(g)) - pairs(len(vals[vals['g']==1])) - pairs(len(vals[vals['g']==0]))
        e0 = 0 if p == 0 else count_inversions(g, 0, len(g)-1, merge_parity_0)[1] / p
        err0.append(e0)
        e1 = 0 if p == 0 else count_inversions(g, 0, len(g)-1, merge_parity_1)[1] / p
        err1.append(e1)
    return err0, err1

def get_all_errs(df, err, window=100, step=10):
    err0,err1 = err(df, window, step)
    errs=[]
    #trends
    r0=[x/len(err0) for x in range(len(err0))]
    r1=[x/len(err1) for x in range(len(err1))]
    errs.append(stats.linregress(r0, y=err0)[0])
    errs.append(stats.linregress(r1, y=err1)[0])
    #correlation
    errs.append(stats.pearsonr(err0,err1)[0])
    #distance
    diffs = np.abs(np.array(err0) - np.array(err1))
    errs.append(np.sum(diffs) / len(err0))
    #significance
    errs.append(stats.ttest_ind(err0,err1)[1])
    return errs

def norm(x):
    x = (x - np.mean(x))/np.std(x)
    
def diagnose_k(df):
     
#     dfs= run(data_g,data_y,data_X)
    
    errs=pd.DataFrame(index=[['trend0','trend1','cor','dist','sig']])
    #statistical parity
    errs['parity'] = get_all_errs(df, sliding_kendall_parity, window=100, step=10)
        
    #Calibration
    errs['cal'] = get_all_errs(df, sliding_kendall_cal2, window=100, step=10)
        
    #Equalized Odds
    errs['eq'] = get_all_errs(df, sliding_kendall_eq, window=100, step=10)
        
    return errs



# original normalized version -- counts all inverted pairs out of only one group
#changed to cal2 which counts all inverted pairs which contain element from that group 
# #calibration
# def merge(h1,h2):
#     count = 0
#     arr = []
#     i=0
#     j=0
#     while i < len(h1) and j< len(h2):
# #         print("compare", h1[i] , h2[j])
#         if h1[i][0] > h2[j][0]:
#             arr.append(h1[i])
#             i+=1
#         else:
# #             print("count",len(h1)-i)
#             count +=(len(h1)-i)
#             arr.append(h2[j])
#             j+=1
            
# #     add any remaining elements
#     while i < len(h1):
#         arr.append(h1[i])
#         i+=1
#     while j < len(h2):
#         arr.append(h2[j])
#         j+=1
#     return arr, count



# def sliding_kendall_calibration(df, window, step):
#     df.sort_values('y_pred', ascending=False, inplace=True)
#     err0=[]
#     err1=[]
#     start=0
#     end=window
#     #     incrementally compute trend
#     while end<len(df):
#         vals = df.iloc[range(start,end)]
#         g0 = np.array(vals[vals['g']==0.].sort_values('y',ascending=False)[['y_pred', 'g']])
#         g1 = np.array(vals[vals['g']==1.].sort_values('y',ascending=False)[['y_pred','g']])
#         p0 = pairs(len(g0))
#         p1 = pairs(len(g1))
#         e0 = 0 if p0 == 0 else count_inversions(g0, 0, len(g0)-1, merge)[1] / p0
#         err0.append(e0)
#         e1 = 0 if p1 == 0 else count_inversions(g1, 0, len(g1)-1, merge)[1] / p1
#         err1.append(e1)
#         start+=step
#         end+=step
#     #get end of rank is needed
#     if(start > len(df)-window):
#         vals = df.iloc[range(len(df)-window,len(df))]
#         g0 = np.array(vals[vals['g']==0.].sort_values('y', ascending=False)[['y_pred', 'g']])
#         g1 = np.array(vals[vals['g']==1.].sort_values('y', ascending=False)[['y_pred', 'g']])
#         p0 = pairs(len(g0))
#         p1 = pairs(len(g1))
#         e0 = 0 if p0 == 0 else count_inversions(g0, 0, len(g0)-1, merge)[1] / p0
#         err0.append(e0)
#         e1 = 0 if p1 == 0 else count_inversions(g1, 0, len(g1)-1, merge)[1] / p1
#         err1.append(e1)
#     return err0, err1


#original versions -- not normalized

# def sliding_kendall_calibration(df, window, step):
#     df.sort_values('y_pred', inplace=True)
#     err0=[]
#     err1=[]
#     start=0
#     end=window
#     while end<len(df):
#         vals = df.iloc[range(start,end)]
#         g0 = np.array(vals[vals['g']==0.].sort_values('y')[['y_pred', 'g']])
#         g1 = np.array(vals[vals['g']==1.].sort_values('y')[['y_pred','g']])
#         err0.append(count_inversions(g0, 0, len(g0)-1, merge)[1])
#         err1.append(count_inversions(g1, 0, len(g1)-1, merge)[1])
#         start+=step
#         end+=step
#     #get end of rank is needed
#     if(start > len(df)-window):
#         vals = df.iloc[range(len(df)-window,len(df))]
#         g0 = np.array(vals[vals['g']==0.].sort_values('y')[['y_pred', 'g']])
#         g1 = np.array(vals[vals['g']==1.].sort_values('y')[['y_pred', 'g']])
#         err0.append(count_inversions(g0, 0, len(g0)-1, merge)[1])
#         err1.append(count_inversions(g1, 0, len(g1)-1, merge)[1])
#     return err0, err1

# def sliding_kendall_eq(df, window, step):
#     df.sort_values('y_pred', inplace=True)
#     err0=[]
#     err1=[]
#     start=0
#     end=window
#     while end<len(df):
#         vals = df.iloc[range(start,end)]
#         g = np.array(vals.sort_values('y')[['y_pred', 'g']])
#         err0.append(count_inversions(g, 0, len(g)-1, merge_0_over)[1])
#         err1.append(count_inversions(g, 0, len(g)-1, merge_1_over)[1])
#         start+=step
#         end+=step
#     #get end of rank is needed
#     if(start > len(df)-window):
#         vals = df.iloc[range(len(df)-window,len(df))]
#         g = np.array(vals.sort_values('y')[['y_pred', 'g']])
#         err0.append(count_inversions(g, 0, len(g)-1, merge_0_over)[1])
#         err1.append(count_inversions(g, 0, len(g)-1, merge_1_over)[1])
#     return err0, err1

# def sliding_kendall_parity(df, window, step):
#     df.sort_values('y_pred', inplace=True)
#     err0=[]
#     err1=[]
#     start=0
#     end=window
#     while end<len(df):
#         vals = df.iloc[range(start,end)]
#         g = np.array(vals[['y_pred', 'g']])
#         err0.append(count_inversions(g, 0, len(g)-1, merge_parity_0)[1])
#         err1.append(count_inversions(g, 0, len(g)-1, merge_parity_1)[1])
#         start+=step
#         end+=step
#     #get end of rank is needed
#     if(start > len(df)-window):
#         vals = df.iloc[range(len(df)-window,len(df))]
#         g = np.array(vals[['y_pred', 'g']])
#         err0.append(count_inversions(g, 0, len(g)-1, merge_parity_0)[1])
#         err1.append(count_inversions(g, 0, len(g)-1, merge_parity_1)[1])
#     return err0, err1