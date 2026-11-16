import numpy as np
from common import load,write_results,write_csv,sample_sd

def kmeans(Z,k):
    step=len(Z)//k; cent=Z[np.array([i*step for i in range(k)])].copy(); labels=np.full(len(Z),-1,dtype=int)
    while True:
        dist=np.sum((Z[:,None,:]-cent[None,:,:])**2,axis=2); new=np.argmin(dist,axis=1)
        changed=not np.array_equal(new,labels); labels=new
        for j in range(k):
            if np.any(labels==j): cent[j]=np.mean(Z[labels==j],axis=0)
        if not changed: break
    wcss=float(np.sum((Z-cent[labels])**2))
    return labels,cent,wcss

def linkage_table():
    names=['E1','E2','E3','E4','E5','E6']
    D=np.array([[0,7.9,7.4,2.9,2.9,6.1],[7.9,0,1.4,5.9,5.3,2.0],[7.4,1.4,0,5.8,4.6,1.3],[2.9,5.9,5.8,0,2.8,4.5],[2.9,5.3,4.6,2.8,0,3.4],[6.1,2.0,1.3,4.5,3.4,0]],float)
    out=[]; cuts={}
    for method in ('single','complete','average'):
        clusters=[[i] for i in range(6)]; heights=[]
        for merge in range(1,6):
            choices=[]
            for i in range(len(clusters)):
                for j in range(i+1,len(clusters)):
                    vals=[D[a,b] for a in clusters[i] for b in clusters[j]]
                    h=min(vals) if method=='single' else max(vals) if method=='complete' else float(np.mean(vals))
                    choices.append((h,i,j))
            h,i,j=min(choices,key=lambda x:x[0]); a=clusters[i]; b=clusters[j]
            if min(a)>min(b): a,b=b,a
            ca='+'.join(names[x] for x in sorted(a)); cb='+'.join(names[x] for x in sorted(b))
            out.append([method,merge,ca,cb,h]); heights.append(h)
            merged=sorted(a+b); clusters=[c for q,c in enumerate(clusters) if q not in (i,j)]+[merged]
        gaps=np.diff(heights); largest=int(np.argmax(gaps)); cuts[method]=6-(largest+1)
    return out,cuts

def main():
    _,_,X,y,val=load(); Z=(X[:,:5]-np.mean(X[:,:5],axis=0))/sample_sd(X[:,:5],axis=0)
    d={}; labels3=None; wc={}
    for k in range(2,7):
        lab,cen,w=kmeans(Z,k); wc[k]=w; d[f'wcss_k{k}']=w
        if k==3: labels3=lab
    elbow=5
    for k in (3,4,5):
        if wc[k]-wc[k+1] < .5*(wc[k-1]-wc[k]): elbow=k; break
    d['elbow_k']=elbow
    # order cluster labels by ascending original X3 mean
    means={j:float(np.mean(X[labels3==j,2])) for j in range(3)}; order=sorted(means,key=means.get)
    seg_rows=[]; seg_index={}
    for label,cluster in zip('ABC',order):
        ix=np.where(labels3==cluster)[0]; seg_index[label]=ix
        d[f'size_segment_{label}']=len(ix)
        seg_rows.append([label,len(ix),*np.mean(X[ix,:5],axis=0),float(np.mean(y[ix])),float(np.mean(val[ix]))])
    write_csv('part3_segments.csv',['segment','size','mean_X1','mean_X2','mean_X3','mean_X4','mean_X5','outcome_rate','mean_value'],seg_rows)
    hrows,cuts=linkage_table(); write_csv('part3_hierarchical.csv',['linkage','merge_no','cluster_a','cluster_b','height'],hrows)
    for k,v in cuts.items(): d[f'{k}_clusters_at_cut']=v
    cov=(Z.T@Z)/(len(Z)-1); eig,vec=np.linalg.eigh(cov); ix=np.argsort(eig)[::-1]; eig=eig[ix]; vec=vec[:,ix]
    for j in range(5):
        if vec[np.argmax(np.abs(vec[:,j])),j]<0: vec[:,j]*=-1
    ratio=eig/np.sum(eig); cum=np.cumsum(ratio)
    d['eigenvalue_sum']=float(np.sum(eig))
    for j in range(5): d[f'eigenvalue_{j+1}']=float(eig[j]); d[f'explained_ratio_{j+1}']=float(ratio[j]); d[f'cumulative_ratio_{j+1}']=float(cum[j])
    d['components_for_80pct']=int(np.where(cum>=.8)[0][0]+1)
    for j in range(5): d[f'pc1_loading_X{j+1}']=float(vec[j,0]); d[f'pc2_loading_X{j+1}']=float(vec[j,1])
    proj=Z@vec[:,:2]
    for i in range(3): d[f'row{i+1}_pc1']=float(proj[i,0]); d[f'row{i+1}_pc2']=float(proj[i,1])
    for label,ix in seg_index.items(): d[f'mean_pc1_segment_{label}']=float(np.mean(proj[ix,0]))
    write_results('part3_results.txt',d); print('Part 3 complete')
if __name__=='__main__': main()
