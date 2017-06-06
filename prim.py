import numpy as np
from joblib import Parallel, delayed
import collections
import pdb
import copy 
def condition_chainer(conditions,action='not'):
    query_string=action+" ((" + string_condition(conditions[0]) + ")"
    for cond in conditions[1:]:
        query_string+=" and (" + string_condition(cond) + ")"
    query_string+=")"
    return query_string
def condense_condition(conditions,existing_reqs):
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
            cat_score=y[x==category]
            score_=cat_score.mean()
            if score_>max_val and cat_score.shape[0]>beta:
                max_val=score_
                max_cat=category
        return [max_val,max_cat,i]
    else:
        min_split=x.quantile(alpha)
        max_split=x.quantile(1-alpha)
        min_split_score = y[(x<min_split)].mean()
        max_split_score = y[(x >= max_split)].mean()
        if min_split_score>=max_split_score and y[(x<min_split)].shape[0]>beta:
            return [min_split_score,[min_split,"min"],i]
        elif min_split_score<max_split_score and y[(x>=max_split)].shape[0]>beta:
            return [max_split_score,[max_split,'max'],i]
        else:
            return [-np.inf,[min_split,"min"],i]
def winning_condition(cand):
    '''Return best conditional split for list cand'''
    max_val=-np.inf
    max_feature=None
    for candidate in cand:
        if candidate[0]>max_val:
            max_val=candidate[0]
            max_feature=candidate
    return(max_feature)
def generate_boxes(conditions):
    boxes_=[]
    for cond in conditions:
        boxes_.append(Box(cond,cond[-1][0]))
    return(boxes_)
class Box:
    def __init__(self,conditions,mean):
        '''For each box, need to build concatenation function to form big not statement'''
        self.conditions=conditions
        self.mean=mean

def box_bounds(conditions_,existing_constraints_):
    supplemental_conditions = []
    for i in conditions_:
        if isinstance(i[1],list):
            if i[1][1] == 'max' and (existing_constraints_[i[2]]['max'] and existing_constraints_[i[2]]['max'] >  i[1][0]):
                supplemental_conditions.append([[0],[existing_constraints_[i[2]]['max'],'min'],i[2]])
            if i[1][1] == 'min' and (existing_constraints_[i[2]]['min'] and existing_constraints_[i[2]]['min'] <  i[1][0]):
                supplemental_conditions.append([i[0],[existing_constraints_[i[2]]['min'],'max'],i[2]])
    return(supplemental_conditions+conditions_)
#                existing_constraints[[i[2]]['min'] = i[1][0]
#    return(existing_constraints)
            
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
            candidates=[]
            candidates=Parallel(n_jobs=-1)(delayed(split_score)(x_view[i],y_view,self.alpha,self.beta,i,self.classes_[i]) for i in x_view.keys())
            winning_filter=winning_condition(candidates)
            if winning_filter in conditions:
                return(conditions)
            conditions.append(winning_filter)
            try:
                x_view=x_view.query(string_condition(winning_filter))
            except:
                x_view=x_view.query(string_condition(winning_filter))
            y_view=y_view[x_view.index]
        return(conditions)
    def fit(self,X,y,n_jobs=-1):
        self.n_jobs=n_jobs
        self.classes_=collections.OrderedDict()
        self.box_conditions=[]
        self.boxes=[]
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
            if len(box_)==0:
                break
            self.box_conditions.append(box_)
            self.boxes.append(Box(self.box_conditions[-1],self.box_conditions[-1][-1][0]))
            x_view=x_view.query(condition_chainer(self.box_conditions[-1]))
            y_view=y_view[x_view.index]
        self.baseline= y.mean()
        return
    def predict(self,X):
        predictions=np.full(X.shape[0],self.baseline)
        indices=[]
        for box_ in self.boxes:
            predictions[X.query(condition_chainer(box_.conditions,"")).index]=box_.mean
            X = X.query(condition_chainer(box_.conditions))
        return(predictions)

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    data=pd.DataFrame({"A":np.random.randint(0,100,1000),"B":np.random.randint(0,100,1000),"C":[np.random.choice(["apples","bananas","oranges"]) for i in range(0,1000)],"D":np.random.normal(size=1000)})
    data['D'][data['C']=='apples']=200
    RGR = PRIM(beta=5) 
    RGR.fit(data[['A','B','C']],data['D'])
    print(set(RGR.predict(data)))
