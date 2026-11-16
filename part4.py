import math, numpy as np
from common import get_split_data,write_results,write_csv,logistic_fit,sigmoid,metrics,knn_predict
from part3 import linkage_table

def entropy(y):
    if len(y)==0: return 0.0
    p=float(np.mean(y)); return -sum(q*math.log2(q) for q in (p,1-p) if q>0)
def gini(y):
    if len(y)==0: return 0.0
    p=float(np.mean(y)); return 1-p*p-(1-p)*(1-p)
def gain(y,mask,fn):
    yes=y[mask]; no=y[~mask]; return fn(y)-len(yes)/len(y)*fn(yes)-len(no)/len(y)*fn(no)

def main():
    ids,X,y,val,tr,te,Z,mu,sd=get_split_data(); yt=y[tr]; yv=y[te]; Xt=Z[tr]; Xv=Z[te]
    d={'root_entropy':entropy(yt),'root_gini':gini(yt)}
    masks=[X[tr,5]==1,X[tr,6]==1,X[tr,2]>.5,X[tr,0]<=24]
    srows=[]
    for name,mask in zip(('S1','S2','S3','S4'),masks):
        yy=yt[mask]; nn=yt[~mask]; srows.append([name,len(yy),int(yy.sum()),len(nn),int(nn.sum()),gain(yt,mask,entropy),gain(yt,mask,gini)])
    write_csv('part4_splits.csv',['split','n_yes','positives_yes','n_no','positives_no','info_gain','gini_gain'],srows)
    besti=max(srows,key=lambda r:r[5])[0]; bestg=max(srows,key=lambda r:r[6])[0]
    d['best_split_by_info_gain']=besti; d['best_split_by_gini_gain']=bestg
    idx={'S1':0,'S2':1,'S3':2,'S4':3}[besti]; mask=masks[idx]
    rate_yes=float(np.mean(yt[mask])); rate_no=float(np.mean(yt[~mask])); d['stump_rate_yes']=rate_yes; d['stump_rate_no']=rate_no
    w,b,_=logistic_fit(Xt,yt,.1,1000,()); logp=sigmoid(Xv@w+b)
    # determine best K exactly as Part 2
    folds=np.arange(len(tr))%5; cv=[]
    for k in range(1,22,2):
        sc=[]
        for fold in range(5):
            qi=np.where(folds==fold)[0]; ri=np.where(folds!=fold)[0]
            kp=knn_predict(Xt[ri],yt[ri],Xt[qi],k); sc.append(metrics(yt[qi],(kp>=(k+1)/(2*k)).astype(int))['f1'])
        cv.append([k,float(np.mean(sc))])
    bestk=sorted(cv,key=lambda r:(-r[1],r[0]))[0][0]; kp=knn_predict(Xt,yt,Xv,bestk)
    stump_mask=[X[te,5]==1,X[te,6]==1,X[te,2]>.5,X[te,0]<=24][idx]
    stump=np.where(stump_mask,rate_yes,rate_no)
    probs=[logp,kp,stump]; preds=[logp>=.3,kp>=(bestk+1)/(2*bestk),stump>=.3]
    hard=(np.sum(np.column_stack(preds),axis=1)>=2); soft=(np.mean(np.column_stack(probs),axis=1)>=.3)
    voting=[]
    for name,p in zip(('logistic','knn','stump','hard_vote','soft_vote'),[preds[0],preds[1],preds[2],hard,soft]):
        m=metrics(yv,p.astype(int)); voting.append([name,m['tp'],m['fp'],m['tn'],m['fn'],m['precision'],m['recall'],m['f1']])
    write_csv('part4_voting.csv',['model','tp','fp','tn','fn','precision','recall','f1'],voting)
    # AdaBoost
    yy=np.array([1,1,1,-1,-1,-1,1,-1]); h1=np.array([1,1,-1,-1,-1,-1,-1,-1]); h2=np.array([-1,1,1,-1,1,-1,1,-1]); weights=np.ones(8)/8; ad=[]; alphas=[]
    for rnd,h in ((1,h1),(2,h2)):
        before=weights.copy(); err=float(np.sum(before[h!=yy])); alpha=.5*math.log((1-err)/err); un=before*np.exp(-alpha*yy*h); weights=un/np.sum(un); alphas.append(alpha)
        for i in range(8): ad.append([rnd,i+1,before[i],un[i],weights[i]])
        d[f'round{rnd}_error']=err; d[f'round{rnd}_alpha']=alpha
    ensemble=np.sign(alphas[0]*h1+alphas[1]*h2); d['ensemble_accuracy']=float(np.mean(ensemble==yy))
    write_csv('part4_adaboost.csv',['round','point','weight_before','weight_unnormalised','weight_after'],ad)
    d['oob_fraction']=(1-1/560)**560
    # parameter counts
    arch=[['mlp_hidden1',(784+1)*16],['mlp_hidden2',(16+1)*16],['mlp_output',(16+1)*10],['cnn_conv1',(5*5*1+1)*6],['cnn_conv2',(5*5*6+1)*16],['cnn_dense1',(5*5*16+1)*32],['cnn_dense2',(32+1)*10]]
    write_csv('part4_architecture.csv',['layer','parameters'],arch)
    d['mlp_total']=sum(r[1] for r in arch[:3]); d['cnn_total']=sum(r[1] for r in arch[3:])
    d['fc_layer_params']=(128*128*3+1)*1024; d['conv_layer_params']=(3*3*3+1)*64
    d['rnn_total']=64*128+128*128+128*16+128+16
    write_results('part4_results.txt',d); print('Part 4 complete')
if __name__=='__main__': main()
