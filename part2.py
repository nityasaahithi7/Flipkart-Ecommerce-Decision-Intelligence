import numpy as np
from common import get_split_data,write_results,write_csv,logistic_fit,sigmoid,metrics,knn_predict,OUT

def main():
    ids,X,y,val,tr,te,Z,mu,sd=get_split_data(); Xt=Z[tr]; Xv=Z[te]; yt=y[tr]; yv=y[te]
    w,b,loss=logistic_fit(Xt,yt,0.1,1000,(0,1,10,100,1000)); prob=sigmoid(Xv@w+b); ptr=sigmoid(Xt@w+b)
    d={'train_rows':len(tr),'train_positives':int(yt.sum()),'test_rows':len(te),'test_positives':int(yv.sum()),
       'loss_epoch_0':loss[0],'loss_epoch_1':loss[1],'loss_epoch_10':loss[10],'loss_epoch_100':loss[100],'loss_epoch_1000':loss[1000]}
    for i,v in enumerate(w,1): d[f'weight_X{i}']=float(v)
    d['bias']=float(b)
    pp=np.clip(prob,1e-15,1-1e-15); d['test_bce']=float(-np.mean(yv*np.log(pp)+(1-yv)*np.log(1-pp)))
    for eta in (0.01,0.1,1.0):
        _,_,ll=logistic_fit(Xt,yt,eta,200,(200,)); d[f'loss_epoch_200_eta_{eta}']=ll[200]
    for i,v in enumerate(w,1): d[f'odds_ratio_X{i}']=float(np.exp(v))
    for tau in (0.5,0.3):
        mm=metrics(yv,(prob>=tau).astype(int))
        for k,v in mm.items():
            if k not in ('tpr','fpr'): d[f'tau_{tau}_{k}']=v
    d['blind_accuracy']=float(np.mean(yv==0))
    threshold_rows=[]
    for tau in np.arange(.1,1,.1):
        m=metrics(yv,(prob>=round(float(tau),1)).astype(int)); threshold_rows.append([round(float(tau),1),m['tp'],m['fp'],m['tn'],m['fn'],m['tpr'],m['fpr']])
    write_csv('part2_thresholds.csv',['tau','tp','fp','tn','fn','tpr','fpr'],threshold_rows)
    folds=np.arange(len(tr))%5; cv=[]
    for k in range(1,22,2):
        scores=[]
        for fold in range(5):
            qi=np.where(folds==fold)[0]; ri=np.where(folds!=fold)[0]
            p=knn_predict(Xt[ri],yt[ri],Xt[qi],k); scores.append(metrics(yt[qi],(p>=(k+1)/(2*k)).astype(int))['f1'])
        cv.append([k,float(np.mean(scores))])
    write_csv('part2_knn_cv.csv',['K','mean_cv_f1'],cv)
    bestk=sorted(cv,key=lambda r:(-r[1],r[0]))[0][0]; knnprob=knn_predict(Xt,yt,Xv,bestk); knnm=metrics(yv,(knnprob>=(bestk+1)/(2*bestk)).astype(int))
    d['knn_best_k']=bestk
    for k,v in knnm.items():
        if k not in ('specificity','tpr','fpr'): d[f'knn_{k}']=v
    costs=[]
    for tau in np.arange(1,20)/20:
        m=metrics(yt,(ptr>=tau).astype(int)); costs.append((50*m['fp']+400*m['fn'],float(tau)))
    cost,tau=min(costs,key=lambda q:(q[0],q[1])); mt=metrics(yv,(prob>=tau).astype(int)); m05=metrics(yv,(prob>=.5).astype(int))
    d.update({'cost_tau':tau,'cost_train_at_tau':cost,'cost_test_at_tau':50*mt['fp']+400*mt['fn'],'cost_test_at_0.5':50*m05['fp']+400*m05['fn'],'cost_test_flag_nobody':400*int(yv.sum())})
    write_results('part2_results.txt',d); print('Part 2 complete')
if __name__=='__main__': main()
