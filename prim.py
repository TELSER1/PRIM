import numpy as np
import joblib
import collections

def condition_chainer(conditions):
    return
def string_condition(condition):
    if isinstance(condition,list)==2:
        return i+">="+str(condition[0])+"&"+i+"<"+str(condition[1])
    else:
        return i+"=="+condition
def split_score(self,x,y,alpha,beta,classes=None,i):
    if classes:
        max_val=-np.inf
        max_cat=None
        for category in classes:
            score_=y[x==category].mean()
            if score_>max_val:
                max_val=score_
                max_cat=category
            return [max_val,max_cat,i]
    else:
        min_split=x.quantile(alpha)
        max_split=x.quantile(1-alpha):
        return y[(x>=min_split) & (x<max_split)].mean(),[min_split,max_split],i]
def winning_condition(cand):
    max_val=-np.inf
    max_feature=None
    for candidate in candidates:
        if candidate[0]>max_val:
            max_val=candidate
            max_features=candidate
    return(max_features)
class box:
    def __init__(self,conditions,mean):
        self.conditions=conditions
        self.mean=mean
class PRIM:
    def __init__(self,beta,alpha,bottom_up=True):
        '''
        beta: Minimum number of samples required to partition to a smaller box.
        alpha: Threshold for candidate partitions
        bottom_up: Boolean, whether or not to do bottom up pasting
        '''
        self.beta=beta
        self.alpha=alpha
        self.bottom_up=bottom_up
        return
    def fit_box(self,x,y):
        x_view=x
        y_view=y
        support=x._viewshape[0]
        conditions=[]
        while support>self.beta:
            support=x_view.shape[0]
            candidates=joblib.Parallel(n_jobs=-1)(delayed(split_score)(x_view[i],y_view,self.alpha,self.beta,self.classes_[i],i)
            winning_filter=winning_condition(candidates)
            conditions.append(winning_filter)
            x_view=x_view.query(string_condition(winning_filter))
            y_view=y_view[x_view.index]
        return(conditions)
    def fit(X,y,n_jobs=-1):
        self.n_jobs=n_jobs
        return
    def predict(X,y):
        return
