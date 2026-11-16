import numpy as np
from common import load,write_results,sample_sd,OUT

def regression(x,y):
    xm=float(np.mean(x)); ym=float(np.mean(y)); slope=float(np.sum((x-xm)*(y-ym))/np.sum((x-xm)**2)); intercept=ym-slope*xm
    fit=intercept+slope*x; tss=float(np.sum((y-ym)**2)); rss=float(np.sum((y-fit)**2)); ess=float(np.sum((fit-ym)**2))
    return slope,intercept,fit,tss,ess,rss,1-rss/tss,float(np.sqrt(rss/len(y))),float(np.mean(np.abs(y-fit)))

def main():
    _,_,X,_,y=load(); n=len(y); x4=X[:,3]
    mean=float(np.mean(y)); sd=float(sample_sd(y)); se=sd/np.sqrt(n)
    d={'n':n,'mean_value':mean,'sd_value':sd,'se_value':se,'ci95_low':mean-1.96*se,'ci95_high':mean+1.96*se,
       'z_stat':(mean-8.0)/se,'reject_h0':'yes' if abs((mean-8.0)/se)>1.96 else 'no',
       'n_required':int(np.ceil((1.96*sd/0.5)**2))}
    slope,inter,fit,tss,ess,rss,r2,rmse,mae=regression(x4,y)
    d.update(reg_slope=slope,reg_intercept=inter,tss=tss,ess=ess,rss=rss,
             tss_equals_ess_plus_rss='yes' if abs(tss-ess-rss)<1e-6*tss else 'no',r2=r2,rmse=rmse,mae=mae)
    z=(x4-np.mean(x4))/sample_sd(x4); m=0.; c=0.; costs={0:float(np.mean(y*y))}
    for epoch in range(1,501):
        e=y-(m*z+c); gm=-2*np.mean(z*e); gc=-2*np.mean(e); m-=0.1*gm; c-=0.1*gc
        if epoch in (1,10,100,500): costs[epoch]=float(np.mean((y-(m*z+c))**2))
    gd_s=m/sample_sd(x4); gd_i=c-m*np.mean(x4)/sample_sd(x4)
    d.update(gd_J0=costs[0],gd_J1=costs[1],gd_J10=costs[10],gd_J100=costs[100],gd_J500=costs[500],gd_slope=gd_s,gd_intercept=gd_i,gd_slope_gap=abs(gd_s-slope))
    order=np.argsort(fit,kind='stable'); resid=y-fit
    d['resid_sd_lower_raw']=float(sample_sd(resid[order[:400]])); d['resid_sd_upper_raw']=float(sample_sd(resid[order[400:]])); d['resid_sd_ratio_raw']=d['resid_sd_upper_raw']/d['resid_sd_lower_raw']
    lx=np.log(x4); ly=np.log(y); ls,li,lfit,ltss,less,lrss,lr2,lrmse,_=regression(lx,ly)
    d.update(loglog_intercept=li,loglog_slope=ls,loglog_r2=lr2,loglog_rmse=lrmse)
    lo=np.argsort(lfit,kind='stable'); lres=ly-lfit
    d['resid_sd_lower_log']=float(sample_sd(lres[lo[:400]])); d['resid_sd_upper_log']=float(sample_sd(lres[lo[400:]])); d['resid_sd_ratio_log']=d['resid_sd_upper_log']/d['resid_sd_lower_log']
    # Normal equation in specified column order: constant, ln(X4), X1,X2,X3,X5,X6,X7
    design=np.column_stack([np.ones(n),np.log(X[:,3]),X[:,0],X[:,1],X[:,2],X[:,4],X[:,5],X[:,6]])
    beta=np.linalg.solve(design.T@design,design.T@ly); pred=design@beta; my=float(np.mean(ly)); mtss=float(np.sum((ly-my)**2)); mrss=float(np.sum((ly-pred)**2)); mr2=1-mrss/mtss
    for i,b in enumerate(beta): d[f'mr_beta_{i}']=float(b)
    d['mr_r2']=mr2; d['mr_adj_r2']=1-(1-mr2)*(n-1)/(n-7-1); d['loglog_adj_r2']=1-(1-lr2)*(n-1)/(n-1-1)
    write_results('part1_results.txt',d)
    print('Part 1 complete')
if __name__=='__main__': main()
