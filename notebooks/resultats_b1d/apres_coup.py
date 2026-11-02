import sys, numpy as np, re, os
SMOKE_TEST=False
src=open(sys.argv[1]).read()
ns={'np':np,'re':re,'os':os,'SMOKE_TEST':False}
exec(src, ns)
POOL=ns['POOL']; BONNE=np.array([a*b for a,b,_,_ in POOL])
def auc(x,y): return float(np.mean(x[:,None]>y[None,:]) + 0.5*np.mean(x[:,None]==y[None,:]))
def pair(x,y): return float(np.mean(x>y)+0.5*np.mean(x==y))
rng=np.random.default_rng(0)
def boot(x,y,B=4000):
    n=len(x); v=[]
    for _ in range(B):
        i=rng.integers(0,n,n); v.append(pair(x[i],y[i]))
    return np.percentile(v,[2.5,97.5])
for m in ['qwen','smol']:
    d={k:np.load(f'{sys.argv[2]}/{m}/{k}.npy') for k in ['idx','W','F','R','score_C','score_W','score_F','score_R']}
    idx=d['idx']; C=BONNE[idx]; a=np.array([POOL[i][0] for i in idx]); b=np.array([POOL[i][1] for i in idx])
    ld=(a%10)*(b%10)%10
    print(f'== {m} n={len(idx)}')
    for k in 'WFR':
        v=d[k]; coh=(v%10==ld)
        print(f'  {k}: dernier chiffre cohérent {coh.mean():.2f} ; |v-C| médian {np.median(np.abs(v-C)):.0f} ; signe + {np.mean(v>C):.2f}')
    for k in 'WFR':
        v=d[k]; coh=(v%10==ld); s=d['score_'+k]; c=d['score_C']
        for nom,msk in (('cohérent',coh),('incohérent',~coh)):
            if msk.sum()>=15:
                ci=boot(s[msk],c[msk]); print(f'  AUC appariée {k} {nom:10s} n={msk.sum():3d} : {pair(s[msk],c[msk]):.3f} [{ci[0]:.3f} ; {ci[1]:.3f}]')
    # R selon le signe par rapport à C (miroir de W)
    for k in 'WR':
        v=d[k]; s=d['score_'+k]; c=d['score_C']
        for nom,msk in (('v>C',v>C),('v<C',v<C)):
            if msk.sum()>=15: print(f'  AUC {k} {nom} n={msk.sum()} : {pair(s[msk],c[msk]):.3f}')
