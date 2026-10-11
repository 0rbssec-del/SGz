"""Architecture à asymétrie critique : contrôle en boucle fermée de A pendant le décodage.

A_hat = moyenne de trois mesures bornées dans [0, 1], calculées sur la
distribution d'échantillonnage q (après température T et top-p) :
  phi1 : entropie normalisée H(q) / log V   (surprise attendue ; cf. Mirostat)
  phi2 : 1 - exp(-KL(q || q_aligné))        (écart modèle libre / aligné)
  phi6 : 1 - exp(-KL(q_avec || q_sans))     (influence de l'information privée)

Deux contrôleurs :
  OriginalController  : portage fidèle du pseudo-code (seules les variables
                        non initialisées le sont), signe et répartition inchangés.
  CorrectedController : signe de commande corrigé (A croît avec p), forme en
                        position, anti-saturation de l'intégrale, T ne bouge que
                        si p est en butée.

Le modèle est un jouet numpy (bigrammes) : aucune dépendance, tourne sur CPU.
Toute classe exposant logits(contexte, prive, aligne) peut le remplacer.
"""
from collections import deque

import numpy as np


def sampling_dist(logits, T, p):
    """Softmax tempérée puis troncature top-p (noyau), renormalisée."""
    z = logits / T
    z = z - z.max()
    q = np.exp(z)
    q /= q.sum()
    order = np.argsort(-q)
    cum = np.cumsum(q[order])
    keep = order[: np.searchsorted(cum, p) + 1]
    out = np.zeros_like(q)
    out[keep] = q[keep]
    return out / out.sum()


def kl(a, b, eps=1e-12):
    m = a > 0
    return float(np.sum(a[m] * (np.log(a[m]) - np.log(b[m] + eps))))


class ToyBigramModel:
    """Logits = W[dernier token] ; version alignée plus piquée ; info privée = biais."""

    def __init__(self, V=200, seed=0, align_gain=1.6, private_strength=2.0):
        rng = np.random.default_rng(seed)
        zipf = -np.log(np.arange(1, V + 1))
        self.W = zipf[rng.permuted(np.tile(np.arange(V), (V, 1)), axis=1)] + rng.normal(0, 0.5, (V, V))
        self.W_al = align_gain * self.W + rng.normal(0, 0.3, (V, V))
        self.private_bias = np.zeros(V)
        self.private_bias[rng.choice(V, V // 10, replace=False)] = private_strength
        self.V = V

    def logits(self, ctx, prive=False, aligne=False):
        row = (self.W_al if aligne else self.W)[ctx[-1]]
        return row + self.private_bias if prive else row


def measure_A(model, ctx, T, p, mesure="echantillon"):
    """Retourne (A_hat, phi1, phi2, phi6, q) ; q est la distribution d'échantillonnage.

    mesure="echantillon" : phi2 et phi6 comparent q (tronquée top-p) à des références
        non tronquées. La troncature gonfle alors les KL (≈ -log masse gardée) : phi1
        monte avec p pendant que phi2 et phi6 descendent.
    mesure="modele" : phi2 et phi6 comparent des distributions non tronquées (p = 1) ;
        ils ne dépendent plus de p, seulement de T et du modèle.
    """
    q = sampling_dist(model.logits(ctx, prive=True), T, p)
    q_ref = q if mesure == "echantillon" else sampling_dist(model.logits(ctx, prive=True), T, 1.0)
    q_al = sampling_dist(model.logits(ctx, prive=True, aligne=True), T, 1.0)
    q_sans = sampling_dist(model.logits(ctx, prive=False), T, 1.0)
    phi1 = -np.sum(q[q > 0] * np.log(q[q > 0])) / np.log(model.V)
    phi2 = 1 - np.exp(-kl(q_ref, q_al))
    phi6 = 1 - np.exp(-kl(q_ref, q_sans))
    return (phi1 + phi2 + phi6) / 3, phi1, phi2, phi6, q


class OriginalController:
    def __init__(self, A_star=0.5):
        self.A_star, self.T, self.p = A_star, 1.0, 0.85
        self.integral = self.erreur_prec = 0.0

    def pid(self, e):
        self.integral += e
        derivee = e - self.erreur_prec
        self.erreur_prec = e
        return 0.3 * e + 0.05 * self.integral + 0.1 * derivee

    def step(self, A_lisse):
        u = self.pid(self.A_star - A_lisse)
        self.T = float(np.clip(self.T, 0.7, 1.3))
        self.p = float(np.clip(1 - (u - 0.15 * np.log(self.T)) / 0.4, 0.5, 0.99))

    def after_sample(self, A_lisse):
        if A_lisse < 0.4:
            self.T *= 1.1
            self.p = min(0.99, self.p + 0.05)
        elif A_lisse > 0.6:
            self.T *= 0.9
            self.p = max(0.5, self.p - 0.05)


class CorrectedController:
    def __init__(self, A_star=0.5, kp=0.3, ki=0.05, kd=0.1, i_max=2.0):
        self.A_star, self.T, self.p0 = A_star, 1.0, 0.85
        self.p = self.p0
        self.kp, self.ki, self.kd, self.i_max = kp, ki, kd, i_max
        self.integral = self.erreur_prec = 0.0

    def step(self, A_lisse):
        e = self.A_star - A_lisse
        derivee = e - self.erreur_prec
        self.erreur_prec = e
        u_libre = self.kp * e + self.ki * (self.integral + e) + self.kd * derivee
        p_libre = self.p0 + u_libre  # signe + : A croît avec p
        self.p = float(np.clip(p_libre, 0.5, 0.99))
        if self.p == p_libre:  # anti-saturation : n'intègre que hors butée
            self.integral = float(np.clip(self.integral + e, -self.i_max, self.i_max))
        else:  # p en butée : T prend le relais, doucement
            self.T = float(np.clip(self.T * (1 + 0.05 * np.sign(e)), 0.7, 1.3))

    def after_sample(self, A_lisse):
        pass


def generate(model, controller, n_tokens=200, window=16, seed=0, mesure="echantillon"):
    rng = np.random.default_rng(seed)
    ctx = [int(rng.integers(model.V))]
    hist = deque(maxlen=window)
    rows = []
    for t in range(n_tokens):
        A_hat, phi1, phi2, phi6, _ = measure_A(model, ctx, controller.T, controller.p, mesure)
        hist.append(A_hat)
        A_lisse = float(np.mean(hist))
        controller.step(A_lisse)
        q = sampling_dist(model.logits(ctx, prive=True), controller.T, controller.p)
        ctx.append(int(rng.choice(model.V, p=q)))
        controller.after_sample(A_lisse)
        rows.append((t, A_hat, A_lisse, phi1, phi2, phi6, controller.T, controller.p))
    return np.array(rows)
