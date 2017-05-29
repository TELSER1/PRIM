import numpy as np
from joblib import Parallel, delayed
import collections
import pdb

def condition_chainer(conditions):
    return
def string_condition(condition_):
    '''Build query string'''
    if isinstance(condition_[1],list):
        return condition_[2]+">="+str(condition_[0])+"&"+condition_[2]+"<"+str(condition_[1])
    else:
        return condition_[2]+"=="+condition_[1]
def split_score(x,y,alpha,beta,i,classes=None):
    '''
    x: Feature vector
    y: Label Column
    alpha: see __init__
    beta: see __init__
    classes: list of possible categorical variable categories
    i: Name of column
    '''
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
        max_split=x.quantile(1-alpha)
        return [y[(x>=min_split) & (x<max_split)].mean(),[min_split,max_split],i]
def winning_condition(cand):
    '''Return best conditional split for list cand'''
    max_val=-np.inf
    max_feature=None
    for candidate in cand:
        if candidate[0]>max_val:
            max_val=candidate
            max_features=candidate
    return(max_features)
class Box:
    def __init__(self,conditions,mean):
        self.conditions=conditions
        self.mean=mean
class PRIM:
    def __init__(self,beta=5,alpha=.05,bottom_up=True):
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
        support=x_view.shape[0]
        conditions=[]
        while support>self.beta:
            support=x_view.shape[0]
            candidates=Parallel(n_jobs=-1)(delayed(split_score)(x_view[i],y_view,self.alpha,self.beta,i,self.classes_[i]) for i in x_view.keys())
            pdb.set_trace()
            winning_filter=winning_condition(candidates)
            conditions.append(winning_filter)
            x_view=x_view.query(string_condition(winning_filter))
            y_view=y_view[x_view.index]
        return(conditions)
    def fit(self,X,y,n_jobs=-1):
        self.n_jobs=n_jobs
        self.classes_=collections.OrderedDict()
        self.box_conditions=[]
        x_view=X
        y_view=y
        for i in X.keys():
            if X[i].dtype == 'object':
                self.classes_[i]=list(set(X[i]))
            else:
                self.classes_[i]=None
        support=x_view.shape[0]
        while support>self.beta:
            support=x_view.shape[0]
            self.box_conditions.append(self.fit_box(x_view, y_view))
            x_view=x_view.query(string_condition(self.box_conditions[-1]))
            y_view=y_view[x_view.index]
            
        self.fit_box(X,y,classes)
        return
    def predict(X,y):
        return

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    data=pd.DataFrame({"A":np.random.randint(0,100,100),"B":np.random.randint(0,100,100),"C":[np.random.choice(["apples","bananas","oranges"]) for i in range(0,100)],"D":np.random.normal(size=100)})
    RGR = PRIM()
    RGR.fit(data[['A','B','C']],data['D'])
