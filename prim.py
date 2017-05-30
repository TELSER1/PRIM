import numpy as np
from joblib import Parallel, delayed
import collections
import pdb

def condition_chainer(conditions):
    query_string="not ((" + string_condition(conditions[0]) + ")"
    for cond in conditions[1:]:
        query_string+=" and (" + string_condition(cond) + ")"
    query_string+=")"
    return query_string
def condense_condition(conditions):
    overall_bounds={}
    for i in conditions:
        if isinstance(i[1],list):
            if i[2] not in overall_bounds.keys():
                overall_bounds[i[2]]={"min":-np.inf,"max":np.inf}
            if i[1][1] == 'max':
                try:
                    overall_bounds[i[2]]['min']=np.max(i[1][0],overall_bounds[i[2]]['min'])
                except:
                    overall_bounds[i[2]]['min']=i[1][0]
            else:
                try:
                    overall_bounds[i[2]]['max']=np.min(i[1][0],overall_bounds[i[2]]['max'])
                except:
                    overall_bounds[i[2]]['max']=i[1][0]
        else:
            try:
                if i[1] not in overall_bounds[i[2]]:
                    overall_bounds[i[2]].append(i[1])
            except:
                overall_bounds[i[2]]=[]
                overall_bounds[i[2]].append(i[1])
    return(overall_bounds)

def string_condition(condition_):
    '''Build query string'''
    quantile_dict={"max":">=","min":"<"}
    if isinstance(condition_[1],list):
        return condition_[2]+quantile_dict[condition_[1][1]]+np.str(condition_[1][0])
    else:
        return condition_[2]+"=="+"'"+condition_[1]+"'"
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
        min_split_score = y[(x<min_split)].mean()
        max_split_score = y[(x >= max_split)].mean()
        if min_split_score>=max_split_score:
            return [min_split_score,[min_split,"min"],i]
        else:
            return [max_split_score,[max_split,'max'],i]
def winning_condition(cand):
    '''Return best conditional split for list cand'''
    max_val=-np.inf
    max_feature=None
    for candidate in cand:
        if candidate[0]>max_val:
            max_val=candidate[0]
            max_features=candidate
    return(max_features)
class Box:
    def __init__(self,conditions,mean):
        self.conditions=conditions
        self.mean=mean
class PRIM:
    def __init__(self,beta=25,alpha=.05,bottom_up=True):
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
            candidates=[]
            candidates=Parallel(n_jobs=-1)(delayed(split_score)(x_view[i],y_view,self.alpha,self.beta,i,self.classes_[i]) for i in x_view.keys())
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
            box_=self.fit_box(x_view, y_view)
            print(condense_condition(box_))
            if len(box_)==0:
                break
            self.box_conditions.append(box_)
            pdb.set_trace()
            x_view=x_view.query(condition_chainer(self.box_conditions[-1]))
            y_view=y_view[x_view.index]
            
        return
    def predict(X,y):
        return

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    data=pd.DataFrame({"A":np.random.randint(0,100,1000),"B":np.random.randint(0,100,1000),"C":[np.random.choice(["apples","bananas","oranges"]) for i in range(0,1000)],"D":np.random.normal(size=1000)})
    data['D'][data['C']=='apples']=200
    print(data['D'].max())
    RGR = PRIM()
    RGR.fit(data[['A','B','C']],data['D'])
