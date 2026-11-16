import csv, math, os
import numpy as np

ROOT=os.path.dirname(os.path.abspath(__file__))
DATA=os.path.join(ROOT,'data','records.csv')
OUT=os.path.join(ROOT,'outputs')
os.makedirs(OUT,exist_ok=True)

def load():
    with open(DATA,newline='') as f: rows=list(csv.DictReader(f))
    names=list(rows[0].keys())
    a=np.array([[float(r[n]) for n in names[1:]] for r in rows],dtype=float)
    # numeric columns in order X1..X7, outcome, value
    return names, [r['record_id'] for r in rows], a[:,:7], a[:,7].astype(int), a[:,8]

def write_results(name,d):
    with open(os.path.join(OUT,name),'w') as f:
        for k,v in d.items():
            if isinstance(v,(float,np.floating)): s=f'{v:.4f}'
            elif isinstance(v,(int,np.integer)): s=str(int(v))
            else: s=str(v)
            f.write(f'{k} = {s}\n')

def write_csv(name,header,rows):
    with open(os.path.join(OUT,name),'w',newline='') as f:
        w=csv.writer(f); w.writerow(header)
        for row in rows:
            w.writerow([f'{x:.4f}' if isinstance(x,(float,np.floating)) else x for x in row])

def sample_sd(x,axis=0): return np.std(x,axis=axis,ddof=1)
def safe_div(a,b): return a/b if b else 0.0
def metrics(y,pred):
    tp=int(np.sum((y==1)&(pred==1))); fp=int(np.sum((y==0)&(pred==1)))
    tn=int(np.sum((y==0)&(pred==0))); fn=int(np.sum((y==1)&(pred==0)))
    precision=safe_div(tp,tp+fp); recall=safe_div(tp,tp+fn)
    return dict(tp=tp,fp=fp,tn=tn,fn=fn,accuracy=safe_div(tp+tn,len(y)),precision=precision,recall=recall,
                f1=safe_div(2*precision*recall,precision+recall),specificity=safe_div(tn,tn+fp),tpr=safe_div(tp,tp+fn),fpr=safe_div(fp,fp+tn))
def sigmoid(z):
    z=np.clip(z,-500,500); return 1/(1+np.exp(-z))
def logistic_fit(X,y,eta=0.1,steps=1000,track=()):
    w=np.zeros(X.shape[1]); b=0.0; losses={}
    def loss():
        p=np.clip(sigmoid(X@w+b),1e-15,1-1e-15)
        return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
    if 0 in track: losses[0]=loss()
    for epoch in range(1,steps+1):
        p=sigmoid(X@w+b); e=p-y
        w-=eta*(X.T@e)/len(y); b-=eta*np.mean(e)
        if epoch in track: losses[epoch]=loss()
    return w,b,losses

def stratified_split(X,y):
    tr=[]; te=[]
    for cls in (0,1):
        ids=np.where(y==cls)[0]; cut=(7*len(ids))//10
        tr.extend(ids[:cut]); te.extend(ids[cut:])
    return np.array(sorted(tr)),np.array(sorted(te))
def knn_predict(refX,refy,qX,k):
    out=[]
    for q in qX:
        dist=np.sqrt(np.sum((refX-q)**2,axis=1))
        ix=np.argsort(dist,kind='stable')[:k]
        out.append(float(np.mean(refy[ix])))
    return np.array(out)
def f1_score(y,p): return metrics(y,p)['f1']
def get_split_data():
    _,ids,X,y,val=load(); tr,te=stratified_split(X,y)
    mu=np.mean(X[tr],axis=0); sd=sample_sd(X[tr],axis=0)
    sd=np.where(sd==0,1,sd)
    return ids,X,y,val,tr,te,(X-mu)/sd,mu,sd
