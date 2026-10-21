"""Audit d'AETHER-SIEVE : les affirmations du banc tiennent-elles ?

Critères fixés avant exécution (seuils de réfutation) :
  C1  Fidélité : corr(organique, purifié) >= 0.98 annoncée.
  C2  Spécificité : sur un flux SANS forçage, le moteur ne doit rien purger
      (énergie purgée < 1 %, corr(organique, purifié) >= 0.99).
  C3  D2 : D2(brut) nettement < D2(purifié) (« écrasé puis restauré »),
      et D2 doit séparer forçage pur / organique pur / bruit blanc.
  C4  Comparaison à un témoin classique (régression harmonique aux
      fréquences détectées, fréquence affinée) : le moteur doit faire au moins aussi bien.
Graines : 20. Moyenne ± écart-type.
"""
import numpy as np
from scipy.fft import rfft, irfft, rfftfreq

from aether_sieve_fixed import AetherSieveEngine, _sliding_manifold_telemetry

FS = 100.0
N = 16384
N_SEEDS = 20
t = np.arange(N) / FS
CARRIERS = [(1.55, 3.75, 0.0, np.sin), (1.20, 9.60, 0.3, np.cos),
            (0.95, 18.25, 0.0, np.sin), (0.70, 31.00, 0.0, np.sin)]


def make(seed, slow_sine=True):
    rng = np.random.default_rng(seed)
    f_ax = rfftfreq(N, d=1.0 / FS)
    env = np.zeros_like(f_ax)
    env[1:] = f_ax[1:] ** -0.45
    pink = irfft(rfft(rng.normal(size=N)) * env, n=N)
    pink /= np.std(pink)
    slow = 0.65 * np.sin(2 * np.pi * 0.22 * t)
    organic = pink + (slow if slow_sine else 0.0)
    forcing = sum(a * fn(2 * np.pi * f * t + ph) for a, f, ph, fn in CARRIERS)
    return organic, forcing, pink, slow


def harmonic_regression(x, freqs_hz):
    """Témoin classique : regroupe les raies détectées, affine chaque fréquence
    (FFT Hann zéro-paddée + interpolation parabolique), soustrait cos/sin par moindres carrés."""
    if len(freqs_hz) == 0:
        return x.copy()
    df = FS / N
    groups, cur = [], [freqs_hz[0]]
    for f in freqs_hz[1:]:
        if f - cur[-1] <= 3 * df:
            cur.append(f)
        else:
            groups.append(cur)
            cur = [f]
    groups.append(cur)
    pad = 8 * N
    spec = np.abs(rfft(x * np.hanning(N), n=pad))
    fz = rfftfreq(pad, d=1.0 / FS)
    cols = []
    for g in groups:
        lo, hi = min(g) - 2 * df, max(g) + 2 * df
        idx = np.where((fz >= lo) & (fz <= hi))[0]
        k = idx[np.argmax(spec[idx])]
        a, b, c = np.log(spec[k - 1:k + 2] + 1e-30)
        f0 = fz[k] + 0.5 * (a - c) / (a - 2 * b + c) * (fz[1] - fz[0])
        cols += [np.cos(2 * np.pi * f0 * t), np.sin(2 * np.pi * f0 * t)]
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, x, rcond=None)
    return x - X @ beta


def corr(a, b):
    return np.corrcoef(a, b)[0, 1]


def d2_mean(x, eng):
    _, _, d2, _ = _sliding_manifold_telemetry(x, eng.window_size, eng.step, eng.tsallis_q,
                                              eng.n_bins, eng.embed_dim, eng.tau)
    return np.mean(d2)


def ms(v):
    v = np.asarray(v)
    return f"{v.mean():.4f} ± {v.std():.4f}"


eng = AetherSieveEngine(fs=FS)
eng.warmup()

rows = {k: [] for k in ["c_raw", "c_eng", "c_eng_vs_pink", "c_hr", "c_hr_vs_pink",
                        "slow_flag", "n_flag", "pump", "d2_raw", "d2_cln",
                        "fp_pump", "fp_nflag", "fp_corr", "fp_corr_noslow", "fp_pump_noslow",
                        "resid_forcing", "slow_kept_eng",
                        "d2_forcing", "d2_pink", "d2_white", "d2_organic"]}

for s in range(N_SEEDS):
    organic, forcing, pink, slow = make(s)
    x = organic + forcing
    rep = eng.process_stream(x)
    clean = rep["signals"]["clean"]
    fr = rep["spectral"]["pumping_freqs_hz"]
    rows["c_raw"].append(corr(organic, x))
    rows["c_eng"].append(corr(organic, clean))
    rows["c_eng_vs_pink"].append(corr(pink, clean))
    rows["n_flag"].append(len(fr))
    rows["slow_flag"].append(np.any(np.abs(fr - 0.22) < 0.02))
    rows["pump"].append(rep["spectral"]["pumping_energy_ratio"])
    rows["d2_raw"].append(np.mean(rep["manifold_raw"]["attractor_d2"]))
    rows["d2_cln"].append(np.mean(rep["manifold_clean"]["attractor_d2"]))
    # Forçage résiduel dans le flux purifié : fraction de la variance du forçage restante
    err = clean - organic
    rows["resid_forcing"].append(np.dot(err, forcing) / np.dot(forcing, forcing))
    # Part de la sinusoïde lente 0.22 Hz conservée par le moteur
    rows["slow_kept_eng"].append(np.dot(clean, slow) / np.dot(slow, slow))
    # Témoin classique : mêmes fréquences détectées, sans 0.22 Hz (on lui laisse l'avantage du doute ? non :
    # on lui donne exactement les détections du moteur, 0.22 Hz compris, pour comparer à armes égales)
    hr = harmonic_regression(x, fr)
    rows["c_hr"].append(corr(organic, hr))
    rows["c_hr_vs_pink"].append(corr(pink, hr))

    # C2 : faux positifs sur flux organique seul
    rep0 = eng.process_stream(organic)
    rows["fp_pump"].append(rep0["spectral"]["pumping_energy_ratio"])
    rows["fp_nflag"].append(len(rep0["spectral"]["pumping_freqs_hz"]))
    rows["fp_corr"].append(corr(organic, rep0["signals"]["clean"]))
    rep00 = eng.process_stream(pink)
    rows["fp_pump_noslow"].append(rep00["spectral"]["pumping_energy_ratio"])
    rows["fp_corr_noslow"].append(corr(pink, rep00["signals"]["clean"]))

    # C3 : D2 discrimine-t-il les régimes ?
    rng = np.random.default_rng(1000 + s)
    rows["d2_forcing"].append(d2_mean(forcing + 1e-3 * rng.normal(size=N), eng))
    rows["d2_pink"].append(d2_mean(pink, eng))
    rows["d2_organic"].append(d2_mean(organic, eng))
    rows["d2_white"].append(d2_mean(rng.normal(size=N), eng))

print(f"Graines : {N_SEEDS}")
print("\n[C1] Fidélité au signal organique")
print(f"  corr(organique, brut)                   {ms(rows['c_raw'])}")
print(f"  corr(organique, purifié moteur)         {ms(rows['c_eng'])}   (annoncé : > 0.98)")
print(f"  corr(organique, témoin rég. harmonique) {ms(rows['c_hr'])}")
print(f"  0.22 Hz organique classée « porteuse »  {np.mean(rows['slow_flag']) * 100:.0f} % des graines")
print(f"  part de la sinusoïde 0.22 Hz conservée  {ms(rows['slow_kept_eng'])}")
print(f"  forçage résiduel dans le purifié        {ms(rows['resid_forcing'])}  (fraction projetée)")
print(f"  bins signalés                           {ms(rows['n_flag'])}")
print(f"  énergie « purgée »                      {ms(rows['pump'])}")
print("\n[C2] Spécificité : flux organique SANS forçage")
print(f"  énergie purgée (rose + 0.22 Hz)         {ms(rows['fp_pump'])}")
print(f"  bins signalés                           {ms(rows['fp_nflag'])}")
print(f"  corr(organique, purifié)                {ms(rows['fp_corr'])}")
print(f"  énergie purgée (rose seul)              {ms(rows['fp_pump_noslow'])}")
print(f"  corr(rose, purifié) (rose seul)         {ms(rows['fp_corr_noslow'])}")
print("\n[C3] Dimension de corrélation D2 (moyenne des fenêtres)")
print(f"  D2 brut (organique + forçage)           {ms(rows['d2_raw'])}")
print(f"  D2 purifié                              {ms(rows['d2_cln'])}")
print(f"  D2 forçage pur (4 sinus, quasi-périodique, vraie dim 4) {ms(rows['d2_forcing'])}")
print(f"  D2 organique pur                        {ms(rows['d2_organic'])}")
print(f"  D2 bruit rose seul                      {ms(rows['d2_pink'])}")
print(f"  D2 bruit blanc                          {ms(rows['d2_white'])}")

# Plafond et témoin hors 0.22 Hz (la raie organique n'est pas un forçage)
c_hr_carriers, c_ceiling = [], []
for s in range(N_SEEDS):
    organic, forcing, pink, slow = make(s)
    x = organic + forcing
    fr = eng.process_stream(x)["spectral"]["pumping_freqs_hz"]
    c_hr_carriers.append(corr(organic, harmonic_regression(x, fr[fr > 1.0])))
    c_ceiling.append(corr(organic, pink))
print("\n[C4] Témoins")
print(f"  plafond si 0.22 Hz retirée : corr(organique, rose)        {ms(c_ceiling)}")
print(f"  rég. harmonique, raies > 1 Hz seulement                   {ms(c_hr_carriers)}")
