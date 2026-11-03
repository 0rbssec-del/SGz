"""B1e, analyse après coup (non pré-enregistrée) : distance en chiffres de W et G ; spécificité et résolution du pilotage."""
import sys, numpy as np
ns = {'np': np, 'SMOKE_TEST': False}
exec(open(sys.argv[1]).read(), ns)                      # banque_b1.py : génération de la banque de B1
BONNE = np.array([a * b for a, b, _, _ in ns['POOL']])
rng = np.random.default_rng(0)
pair = lambda x, y: float(np.mean(x > y) + 0.5 * np.mean(x == y))
def boot(x, y, B=4000):
    n = len(x); v = [pair(x[i], y[i]) for i in (rng.integers(0, n, n) for _ in range(B))]
    return np.percentile(v, [2.5, 97.5])
def ham(a, b):
    a, b = str(a), str(b)
    return sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))
for m in ['qwen', 'smol']:
    L = lambda k: np.load(f'{sys.argv[2]}/{m}/{k}.npy')
    idx = L('idx'); C = BONNE[idx]; W, G = L('W'), L('G')
    sc, sw, sg = L('score_C'), L('score_W'), L('score_G')
    hw = np.array([ham(c, w) for c, w in zip(C, W)]); hg = np.array([ham(c, g) for c, g in zip(C, G)])
    print(f'== {m} n={len(idx)} ; chiffres faux : W moyenne {hw.mean():.2f} (1 chiffre : {np.mean(hw == 1):.2f}) ; '
          f'G moyenne {hg.mean():.2f} (1 chiffre : {np.mean(hg == 1):.2f})')
    for nom, s, h in (('W', sw, hw), ('G', sg, hg)):
        for k, msk in (('1 chiffre', h == 1), ('≥ 2 chiffres', h >= 2)):
            if msk.sum() >= 20:
                ci = boot(s[msk], sc[msk])
                print(f'  AUC appariée {nom} {k:13s} n={msk.sum():3d} : {pair(s[msk], sc[msk]):.3f} [{ci[0]:.3f} ; {ci[1]:.3f}]')
    # pilotage
    bW, bC = L('W_base_M'), L('C_base_M')
    dW, dC = L('W_erreur_+4_M') - bW, L('C_erreur_+4_M') - bC
    print(f'  pilotage α=4 : ΔM runs W {dW.mean():+.4f}, runs C {dC.mean():+.4f} ; part de ΔM nuls : W {np.mean(dW == 0):.2f}, '
          f'C {np.mean(dC == 0):.2f} ; corrélation par problème ΔM(W)–ΔM(C) {np.corrcoef(dW, dC)[0, 1]:+.2f}')
    print(f'  |M| de base médian {np.median(np.abs(bW)):.2f} nat ; ΔM / |M| médian {np.median(dW / np.abs(bW)):+.4f} ; '
          f'plus petit écart non nul de M entre problèmes {np.min(np.diff(np.unique(bW))):.4f} nat')
    r = L('W_remplace_L18_M') - bW
    print(f'  remplacement de tout l\'état L18 par celui du run C : ΔM moyen {r.mean():+.3f} nat, maximum {r.max():+.3f}')
