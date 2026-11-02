"""Fabrique sgz_pilotage_b1e.py à partir de sgz_compare_b1d.py : mêmes problèmes, même position, même sonde ;
partie 1 = produit étranger à dernier chiffre cohérent ; partie 2 = pilotage par la direction de l'erreur."""
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()


def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (s.count(a), a[:100])
    s = s.replace(a, b)


HEAD = r'''# %% [markdown]
# # SGZ B1e — Le signal d'erreur : comparaison au-delà du dernier chiffre ? Et le modèle s'en sert-il ?
#
# **Point de départ (B1d, 2 nov.).** Une sonde entraînée sur C (bonne réponse) contre W (erreur du modèle), à la position
# où la recopie diverge, détecte aussi un produit étranger F (0,935 et 0,873) : verdict COMPARAISON. Mais F rompait la
# cohérence du dernier chiffre (dernier chiffre de v ≠ dernier chiffre de (a mod 10)(b mod 10)) dans 99 à 100 % des cas, et
# la sonde de surface séparait F parfaitement. Chez Qwen3, les erreurs miroir à dernier chiffre incohérent donnaient 0,944,
# comme F (après coup). La comparaison démontrée peut donc se réduire à un contrôle du dernier chiffre.
#
# **Partie 1 — comparaison au-delà du dernier chiffre.** Même dispositif que B1d, avec deux changements fixés d'avance :
# - seuls les problèmes dont l'erreur W a un dernier chiffre **cohérent** sont gardés (la sonde ne peut pas apprendre le
#   dernier chiffre) ;
# - **G, produit étranger cohérent** : vrai produit d'autres facteurs de la banque, même longueur et même début que C,
#   **même dernier chiffre que C**, le plus proche de W, même recopie que C jusqu'à la position mesurée.
#
# G est faux pour ce problème, mais a la forme d'un produit correct et le bon dernier chiffre. Comparaison prédit : G est
# détecté comme W. Forme (la sonde reconnaît un nombre qui n'est pas un produit plausible) prédit : G ressemble à C.
#
# **Partie 2 — le modèle se sert-il de ce signal ?** (idée de l'assistant après B1d, demandée par Erick.) À la même position,
# la sortie ne doute pas (B1c : entropie ~1e-4) : le modèle recopie l'erreur. On ajoute à l'état de la couche L18, à la
# dernière position seulement, la direction de l'erreur, et on regarde si le modèle se met à recopier la bonne réponse
# (« corriger ») ou à hésiter.
# - **Direction** : différence des moyennes W − C des états à L18 (direction des moyennes, préférée aux poids de sonde pour
#   les interventions par Marks et Tegmark 2023, non revérifié ici), estimée sur l'autre moitié des problèmes (validation
#   croisée en deux moitiés). Amplitude α × d, où d est la distance entre les deux moyennes.
# - **Mesures** au jeton suivant (le jeton où les recopies de C et de W diffèrent) : M = log p(jeton de C) − log p(jeton de W)
#   sur les runs W (monte si le modèle corrige) ; D = −log p(jeton propre) (monte si le modèle hésite ; à sortie quasi
#   déterministe, D est presque la divergence de Kullback-Leibler à la sortie sans pilotage).
# - **Témoins** : 16 directions aléatoires de même norme (le pilotage doit battre une perturbation quelconque) ; témoin du
#   crochet : remplacer la sortie de la dernière couche du run W par celle du run C doit faire monter M de plus de 1 nat.
#
# ## Critères (fixés avant toute exécution ; AUC appariée = P(score(x) > score(C)) sur les problèmes ; bootstrap, B = 1000)
#
# **Partie 1, par modèle** (couche L18 fixée d'avance, sonde entraînée sur C contre W seulement, 5 plis par problème) :
# - **Reproduction** : borne basse de l'AUC appariée W contre C > 0,55 ; sinon NON INTERPRÉTABLE.
# - **Surface** : sonde de surface (entraînée sur C contre W) appliquée à G, borne haute < 0,60 ; sinon NON INTERPRÉTABLE
#   (G encore lisible en surface).
# - **COMPARAISON** si la borne basse de l'AUC appariée G contre C > 0,55 ; **FORME** si sa borne haute < 0,55 et si
#   l'écart AUC(W) − AUC(G) a une borne basse > 0,05 ; **INDÉTERMINÉ** sinon.
# - Synthèse, thèse « l'état caché compare la valeur écrite au produit vrai au-delà du dernier chiffre » : SOUTENUE si
#   COMPARAISON pour les deux modèles ; RÉFUTÉE si FORME pour les deux ; DÉPEND DU MODÈLE si l'un et l'autre ; sinon
#   INDÉTERMINÉE.
#
# **Partie 2, par modèle** (runs W des 240 premiers problèmes retenus, α = 4 fixé d'avance ; ΔM, ΔD = écart au run sans
# pilotage, par problème ; écart à la moyenne des 16 directions aléatoires, IC95 par bootstrap sur les problèmes) :
# - témoin du crochet raté (ΔM moyen ≤ 1 nat) → NON INTERPRÉTABLE ;
# - **CORRIGE** si ΔM moyen de la direction de l'erreur > celui de chacune des 16 directions aléatoires et si la borne basse
#   de son écart à leur moyenne > 0 ;
# - sinon **PERTURBE** si la même condition vaut pour ΔD ;
# - sinon **INERTE** si les IC95 des deux écarts contiennent 0 ; sinon **INDÉTERMINÉ**.
# - Synthèse, thèse « le modèle se sert de la direction de l'erreur pour corriger » : SOUTENUE si CORRIGE pour les deux ;
#   RÉFUTÉE si chacun est INERTE ou PERTURBE ; DÉPEND DU MODÈLE si CORRIGE pour un seul et l'autre INERTE ou PERTURBE ;
#   sinon INDÉTERMINÉE.
# - **Descriptifs** : dose α = 1, 2, 4, 8, 16 et α = −4 ; direction des poids de la sonde logistique (α = 4) ; remplacement de
#   l'état L18 par celui du run C (prise de la couche, ordre de grandeur d'un effet possible) ; part des runs dont le jeton le plus probable devient celui de C ;
#   même pilotage sur les runs C (M et D vus depuis C) ; norme ajoutée rapportée à la norme médiane de l'état.
# - Témoins positifs synthétiques pour chaque règle.
#
# ## Prédictions de l'assistant (écrites avant exécution) [spéculation]
#
# Partie 1 : COMPARAISON 40 %, FORME 30 % par modèle ; synthèse SOUTENUE 25 %, RÉFUTÉE 15 %.
# Partie 2 : CORRIGE 10 %, PERTURBE 30 %, INERTE 50 % par modèle ; synthèse SOUTENUE 5 %, RÉFUTÉE 55 %.
#
# ## Limites connues
#
# - Partie 1 : G est le produit le plus proche de W qui garde le dernier chiffre ; d'autres traits de forme (somme des
#   chiffres, divisibilité par 3 ou 9) ne sont pas contrôlés. La sonde de surface ne contient que les derniers chiffres, le
#   premier chiffre et la longueur.
# - Partie 2 : une seule couche, une seule position ; le signal pourrait servir ailleurs (positions de la valeur écrite,
#   étapes suivantes du calcul). Un résultat INERTE dit « pas utilisé à cet endroit pour la recopie », pas « jamais utilisé ».
#   Recopier l'erreur est d'ailleurs la bonne conduite pour une tâche de recopie : CORRIGE serait une surprise.
# - M part de très bas (recopie quasi certaine) : toute perturbation qui aplatit la loi le fait monter. C'est pourquoi
#   la direction de l'erreur n'est jugée que contre des directions aléatoires de même norme, jamais contre zéro.
#   Si la norme ajoutée est petite devant celle de l'état, INERTE peut venir d'une dose trop faible : la dose 16 et le
#   remplacement de l'état L18 donnent l'échelle.
#
'''
i0 = s.index("# %%\nimport gc")
s = HEAD + s[i0:]

rep('"/kaggle/working/sgz_b1d" if os.path.isdir("/kaggle/working") else "sgz_b1d_out"',
    '"/kaggle/working/sgz_b1e" if os.path.isdir("/kaggle/working") else "sgz_b1e_out"')
rep('"resume_b1d.json"', '"resume_b1e.json"')
rep("SEUIL_AUC, ECART_MIN = 0.55, 0.05\n",
    "SEUIL_AUC, ECART_MIN = 0.55, 0.05\n"
    "SEUIL_SURF = 0.60                                # partie 1 : G ne doit plus être lisible en surface\n"
    "N_DIR, N_ALEA, ALPHAS, ALPHA_REF = 240, 16, (1, 2, 4, 8, 16), 4   # partie 2 : problèmes pilotés, directions témoins, doses\n"
    "SEUIL_CROCHET = 1.0                              # nat : le remplacement de la dernière couche doit déplacer M\n")
rep("    BOOT_N = 50\n", "    BOOT_N = 50\n    N_DIR, N_ALEA = 30, 4\n")

# ---- Règles de B1e et témoins positifs (remplacent celles de B1d)
j1 = s.index("# ---------- Règles de B1d et témoins positifs ----------")
j2 = s.index("# Produits de facteurs dans les plages de la banque")
REGLES = r'''# ---------- Règles de B1e et témoins positifs ----------
def apparie(sx, sc, rng, sy=None):
    """AUC appariée P(sx > sc) (+ ½ égalités) sur les problèmes, IC95 par bootstrap ; avec sy, écart AUC(sx) − AUC(sy)."""
    f = lambda a, b: float(np.mean(a > b) + 0.5 * np.mean(a == b))
    n = len(sx)
    bs, bd = [], []
    for _ in range(BOOT_N):
        ix = rng.integers(0, n, n)
        bs.append(f(sx[ix], sc[ix]))
        if sy is not None:
            bd.append(f(sx[ix], sc[ix]) - f(sy[ix], sc[ix]))
    out = {"auc": f(sx, sc), "ci": ci(bs)}
    if sy is not None:
        out.update({"ecart": f(sx, sc) - f(sy, sc), "ecart_ci": ci(bd)})
    return out


def regle_e1(a_w, a_g, surf_g, interpretable=True):
    """a_w : W contre C (avec l'écart AUC(W) − AUC(G)) ; a_g : G contre C ; surf_g : sonde de surface appliquée à G."""
    if not interpretable:
        return "NON INTERPRÉTABLE"
    if a_w["ci"][0] <= SEUIL_AUC:
        return "NON INTERPRÉTABLE (sonde de B1c non reproduite)"
    if surf_g["ci"][1] >= SEUIL_SURF:
        return "NON INTERPRÉTABLE (G lisible en surface)"
    if a_g["ci"][0] > SEUIL_AUC:
        return "COMPARAISON"
    if a_g["ci"][1] < SEUIL_AUC and a_w["ecart_ci"][0] > ECART_MIN:
        return "FORME"
    return "INDÉTERMINÉ"


def ecart_alea(dx, dr, rng):
    """dx : (n,) effet de la direction testée ; dr : (k, n) effets des directions aléatoires. Moyenne de dx, maximum des
    moyennes aléatoires, écart dx − moyenne aléatoire (par problème) et son IC95 par bootstrap sur les problèmes."""
    e = dx - dr.mean(0)
    n = len(e)
    bs = [float(np.mean(e[rng.integers(0, n, n)])) for _ in range(BOOT_N)]
    return {"moy": float(np.mean(dx)), "max_alea": float(np.max(dr.mean(1))), "min_alea": float(np.min(dr.mean(1))),
            "ecart": float(np.mean(e)), "ci": ci(bs)}


def regle_e2(eM, eD, crochet, interpretable=True):
    """eM, eD : sorties de ecart_alea pour ΔM et ΔD ; crochet : ΔM moyen du témoin du crochet."""
    if not interpretable:
        return "NON INTERPRÉTABLE"
    if not crochet > SEUIL_CROCHET:
        return "NON INTERPRÉTABLE (témoin du crochet raté)"
    if eM["moy"] > eM["max_alea"] and eM["ci"][0] > 0:
        return "CORRIGE"
    if eD["moy"] > eD["max_alea"] and eD["ci"][0] > 0:
        return "PERTURBE"
    if eM["ci"][0] <= 0 <= eM["ci"][1] and eD["ci"][0] <= 0 <= eD["ci"][1]:
        return "INERTE"
    return "INDÉTERMINÉ"


def synthese_e1(v, interpretable=True):
    ms = list(v)
    if not interpretable:
        return "NON INTERPRÉTABLE"
    if not ms:
        return "INDÉTERMINÉE"
    if all(v[m] == "COMPARAISON" for m in ms):
        return "SOUTENUE"
    if all(v[m] == "FORME" for m in ms):
        return "RÉFUTÉE"
    if set(v.values()) == {"COMPARAISON", "FORME"}:
        return "DÉPEND DU MODÈLE"
    return "INDÉTERMINÉE"


def synthese_e2(v, interpretable=True):
    ms = list(v)
    if not interpretable:
        return "NON INTERPRÉTABLE"
    if not ms:
        return "INDÉTERMINÉE"
    non = {"INERTE", "PERTURBE"}
    if all(v[m] == "CORRIGE" for m in ms):
        return "SOUTENUE"
    if all(v[m] in non for m in ms):
        return "RÉFUTÉE"
    if "CORRIGE" in v.values() and all(v[m] in non | {"CORRIGE"} for m in ms):
        return "DÉPEND DU MODÈLE"
    return "INDÉTERMINÉE"


rng_d = np.random.default_rng(SEED + 11)
n_d = 400
sc_d = rng_d.normal(0, 1, n_d)
sw_d = rng_d.normal(0.8, 1, n_d)
surf_ok = {"auc": 0.5, "ci": (0.47, 0.53)}
surf_ko = {"auc": 0.9, "ci": (0.85, 0.95)}


def cas1(sw, sg, surf=surf_ok):
    a_w = apparie(sw, sc_d, rng_d, sg)
    return regle_e1(a_w, apparie(sg, sc_d, rng_d), surf)


def cas2(m_dir, m_alea, d_dir, d_alea, crochet=5.0):
    """Effets synthétiques : moyennes par direction, bruit par problème."""
    k = 8
    bruit = lambda m, *forme: m + (lambda z: z - z.mean(axis=-1, keepdims=True))(rng_d.normal(0, 1, forme))  # moyenne exacte
    dM, dMr = bruit(m_dir, n_d), bruit(m_alea, k, n_d)
    dD, dDr = bruit(d_dir, n_d), bruit(d_alea, k, n_d)
    return regle_e2(ecart_alea(dM, dMr, rng_d), ecart_alea(dD, dDr, rng_d), crochet)


checks_e = [("B1e comparaison", cas1(sw_d, rng_d.normal(0.8, 1, n_d)), "COMPARAISON"),
            ("B1e forme", cas1(sw_d, rng_d.normal(0.0, 1, n_d)), "FORME"),
            ("B1e surface", cas1(sw_d, rng_d.normal(0.8, 1, n_d), surf_ko), "NON INTERPRÉTABLE (G lisible en surface)"),
            ("B1e non reproduite", cas1(rng_d.normal(0, 1, n_d), rng_d.normal(0.8, 1, n_d)),
             "NON INTERPRÉTABLE (sonde de B1c non reproduite)"),
            ("B1e indéterminé", cas1(sw_d, rng_d.normal(0.15, 1, n_d)), "INDÉTERMINÉ"),
            ("pilotage corrige", cas2(0.6, 0.0, 0.0, 0.0), "CORRIGE"),
            ("pilotage perturbe", cas2(0.0, 0.0, 0.6, 0.0), "PERTURBE"),
            ("pilotage inerte", cas2(0.0, 0.0, 0.0, 0.0), "INERTE"),
            ("pilotage crochet raté", cas2(0.6, 0.0, 0.0, 0.0, crochet=0.2), "NON INTERPRÉTABLE (témoin du crochet raté)"),
            ("synthèse 1 soutenue", synthese_e1({"a": "COMPARAISON", "b": "COMPARAISON"}), "SOUTENUE"),
            ("synthèse 1 réfutée", synthese_e1({"a": "FORME", "b": "FORME"}), "RÉFUTÉE"),
            ("synthèse 1 indéterminée", synthese_e1({"a": "COMPARAISON", "b": "INDÉTERMINÉ"}), "INDÉTERMINÉE"),
            ("synthèse 2 soutenue", synthese_e2({"a": "CORRIGE", "b": "CORRIGE"}), "SOUTENUE"),
            ("synthèse 2 réfutée", synthese_e2({"a": "INERTE", "b": "PERTURBE"}), "RÉFUTÉE"),
            ("synthèse 2 dépend", synthese_e2({"a": "CORRIGE", "b": "INERTE"}), "DÉPEND DU MODÈLE"),
            ("synthèse 2 indéterminée", synthese_e2({"a": "CORRIGE", "b": "INDÉTERMINÉ"}), "INDÉTERMINÉE")]
for nom, got, att in checks_e:
    print(f"[TÉMOIN POSITIF] {nom:<34} : {str(got):<58} (attendu {att})")
POSITIVE_CONTROL_OK = POSITIVE_CONTROL_OK and all(g == e for _, g, e in checks_e)
print(f"[TÉMOIN POSITIF] B1e {'RÉUSSI' if POSITIVE_CONTROL_OK else 'RATÉ : verdicts en NON INTERPRÉTABLE'}")

'''
s = s[:j1] + REGLES + s[j2:]

# ---- Passe pilotée (après passe)
k1 = s.index("# %%\n# ---------- Boucle principale ----------")
PILOTE = r'''@torch.no_grad()
def passe_dirige(g, seqs, autre, propre, couche, vecs=None, capture=False):
    """Comme passe, sans états cachés : ajoute vecs[i] à la sortie du bloc « couche » (indice des états cachés : 1 = sortie
    du premier bloc) à la dernière position. Renvoie log p(autre), log p(propre), jeton le plus probable et, avec capture,
    la sortie du bloc à la dernière position (après ajout)."""
    bloc = g.model.model.layers[couche - 1]
    head = g.model.get_output_embeddings()
    n = len(seqs)
    lp_a, lp_p, top = np.zeros(n), np.zeros(n), np.zeros(n, np.int64)
    caps, etat = [], {"v": None, "cap": None}

    def crochet(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if etat["v"] is not None:
            h[:, -1, :] += etat["v"].to(h.dtype)
        if capture:
            etat["cap"] = h[:, -1, :].float().cpu().numpy()
        return out

    hd = bloc.register_forward_hook(crochet)
    try:
        for b0 in range(0, n, g.budget):
            loc = list(range(b0, min(n, b0 + g.budget)))
            ids, attn, _ = g.left_pad([seqs[i] for i in loc])
            pos = (attn.cumsum(-1) - 1).clamp(min=0)
            etat["v"] = None if vecs is None else torch.tensor(np.stack([vecs[i] for i in loc]), device=g.dev)
            out = g.model.model(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=False)
            z = head(out.last_hidden_state[:, -1, :].to(head.weight.device)).float()
            lp = torch.log_softmax(z.double(), dim=-1)
            for j, i in enumerate(loc):
                lp_a[i], lp_p[i], top[i] = float(lp[j, autre[i]]), float(lp[j, propre[i]]), int(torch.argmax(lp[j]))
            if capture:
                caps.append(etat["cap"])
            del out
    finally:
        hd.remove()
    return lp_a, lp_p, top, (np.concatenate(caps) if capture else None)

'''
s = s[:k1] + PILOTE + s[k1:]

# ---- Corps de traiter : sélection, partie 1, partie 2
i1 = s.index('        seqs = {k: [] for k in "CWFR"}')
i2 = s.index('        print(f"[DURÉE] {label}')
CORPS = r'''        seqs = {k: [] for k in "CWG"}
        tokj = {k: [] for k in "CWG"}                 # jeton de la recopie à la position mesurée (le suivant)
        keep, jdiv, vals_ = [], [], {k: [] for k in "WG"}
        ko_tok = ko_div = ko_g = ko_wi = 0
        for i, w in zip(idx0, w0):
            C, W = int(BONNE[i]), int(w)
            ud = (POOL[i][0] % 10) * (POOL[i][1] % 10) % 10
            if W % 10 != ud:                            # erreur à dernier chiffre incohérent : exclue (fixé d'avance)
                ko_wi += 1
                continue
            rc, rw = recopie(tok, PROMPTS[i], enc[i], C), recopie(tok, PROMPTS[i], enc[i], W)
            if rc is None or rw is None:
                ko_tok += 1
                continue
            (fc, cc), (fw, cw) = rc, rw
            j = next((k for k in range(min(len(cc), len(cw))) if cc[k] != cw[k]), None)
            if j is None:
                ko_div += 1
                continue
            nd = len(re.sub(r"\D", "", tok.decode(cc[:j])))         # chiffres communs avant la position mesurée
            L_ = len(str(C))
            meme = PRODUITS[(PRODUITS >= 10 ** (L_ - 1)) & (PRODUITS < 10 ** L_) & (PRODUITS % 10 == ud)]
            if nd:
                meme = meme[meme // 10 ** (L_ - nd) == C // 10 ** (L_ - nd)]
            meme = meme[np.argsort(np.abs(meme - W), kind="stable")][:40]
            f = chercher(i, C, W, cc, j, meme)
            if f is None:
                ko_g += 1
                continue
            seqs["C"].append(enc[i] + fc + cc[:j])
            seqs["W"].append(enc[i] + fw + cw[:j])
            seqs["G"].append(enc[i] + f[1] + f[2][:j])
            tokj["C"].append(int(cc[j]))
            tokj["W"].append(int(cw[j]))
            tokj["G"].append(int(f[2][j]))
            keep.append(i)
            jdiv.append(j)
            vals_["W"].append(W)
            vals_["G"].append(f[0])
        n = len(keep)
        print(f"[SÉLECTION] {label} : {len(idx0)} problèmes ({origine}) ; erreur à dernier chiffre incohérent (exclue) {ko_wi} ; "
              f"retenus {n} ; découpage incohérent {ko_tok} ; recopie sans divergence {ko_div} ; sans produit étranger cohérent {ko_g}")
        a["selection"] = {"origine": origine, "problemes": len(idx0), "retenus": n, "w_incoherent": ko_wi,
                          "decoupage_incoherent": ko_tok, "sans_divergence": ko_div, "sans_etranger_coherent": ko_g}
        if n < N_MIN:
            a["verdict"] = a["verdict_pilotage"] = "NON TESTABLE (trop peu de problèmes)"
            ALL[label] = a
            print(f"[VERDICT] {label} : {a['verdict']}")
            raise TropPeu
        if ko_tok > 0.1 * max(1, len(idx0)):
            raise RuntimeError("découpage en jetons incohérent pour plus de 10 % des problèmes : positions non fiables")
        C_ = np.array([int(BONNE[i]) for i in keep], np.int64)
        assert all(int(v) % 10 == int(c) % 10 for v, c in zip(vals_["G"], C_)), "G à dernier chiffre incohérent"
        rel = {k: np.abs(np.array(vals_[k]) - C_) / C_ for k in "WG"}
        print(f"[VALEURS] {label} écart relatif médian à C : W {np.median(rel['W']):.4f}, G {np.median(rel['G']):.4f} ; "
              f"exemple : C {C_[0]}, W {vals_['W'][0]}, G {vals_['G'][0]}")
        t2 = time.time()
        tous = seqs["C"] + seqs["W"] + seqs["G"]
        zero = [0] * len(tous)
        H, _, _, _, _ = passe(g, tous, zero, zero)
        Hs = {k: H[m * n:(m + 1) * n] for m, k in enumerate("CWG")}
        print(f"[MESURE] {label} passe ({3 * n} textes) : {time.time() - t2:.0f} s", flush=True)
        nL = H.shape[1] - 1
        couche = min(COUCHE, nL)

        # ---------- Partie 1 : produit étranger cohérent ----------
        def scores(c):
            """Sonde entraînée sur C contre W (plis par problème), scores hors échantillon de C, W, G."""
            out = {k: np.zeros(n) for k in "CWG"}
            for tr, te in GroupKFold(PLIS).split(np.zeros(n), np.zeros(n), np.arange(n)):
                X = np.concatenate([Hs["C"][tr, c], Hs["W"][tr, c]])
                yy = np.r_[np.zeros(len(tr), int), np.ones(len(tr), int)]
                sc = ajuster(X, yy)
                for k in "CWG":
                    out[k][te] = sc(Hs[k][te, c])
            return out

        t3 = time.time()
        rngB = np.random.default_rng(SEED + 3)
        sc0 = scores(couche)
        a_w = apparie(sc0["W"], sc0["C"], rngB, sc0["G"])
        a_g = apparie(sc0["G"], sc0["C"], rngB)
        pb = [POOL[i] for i in keep]
        Fs = {k: np.stack([surface(int(p[0]), int(p[1]), int(v)) for p, v in zip(pb, vv)])
              for k, vv in (("C", C_), ("W", vals_["W"]), ("G", vals_["G"]))}
        ss = {k: np.zeros(n) for k in "CWG"}
        for tr, te in GroupKFold(PLIS).split(np.zeros(n), np.zeros(n), np.arange(n)):
            sc = ajuster(np.concatenate([Fs["C"][tr], Fs["W"][tr]]), np.r_[np.zeros(len(tr), int), np.ones(len(tr), int)])
            for k in "CWG":
                ss[k][te] = sc(Fs[k][te])
        surf_w, surf_g = apparie(ss["W"], ss["C"], rngB), apparie(ss["G"], ss["C"], rngB)
        verdict = regle_e1(a_w, a_g, surf_g, ok)
        par_couche = {}
        for c in range(1, nL + 1):
            s_ = scores(c)
            f_ = lambda x, y_: float(np.mean(x > y_) + 0.5 * np.mean(x == y_))
            par_couche[c] = {"W": f_(s_["W"], s_["C"]), "G": f_(s_["G"], s_["C"])}
        print(f"[MESURE] {label} sondes : {time.time() - t3:.0f} s", flush=True)
        fa = lambda r: f"{r['auc']:.3f} IC95 [{r['ci'][0]:.3f} ; {r['ci'][1]:.3f}]"
        print(f"\n{'=' * 100}\n[RÉSULTATS 1] {label} : {n} problèmes, couche L{couche} (fixée d'avance)")
        print(f"[REPRODUCTION] {label} W contre C (dernier chiffre cohérent) : AUC appariée {fa(a_w)}")
        print(f"[G] {label} produit étranger cohérent contre C : AUC appariée {fa(a_g)} ; écart W − G {a_w['ecart']:+.3f} IC95 "
              f"[{a_w['ecart_ci'][0]:+.3f} ; {a_w['ecart_ci'][1]:+.3f}]")
        print(f"[SURFACE] {label} sonde de surface (entraînée sur C contre W) : W {fa(surf_w)} ; G {fa(surf_g)}")
        print(f"[COUCHES] {label} AUC appariée par couche, W | G : " + " ".join(
            f"L{c} {v['W']:.2f}|{v['G']:.2f}" for c, v in par_couche.items()))
        print(f"[VERDICT 1] {label} : {verdict}")
        a.update({"n": n, "couche": couche, "verdict": verdict, "W": a_w, "G": a_g,
                  "surface": {"W": surf_w, "G": surf_g}, "par_couche": par_couche,
                  "ecart_relatif_median": {k: float(np.median(rel[k])) for k in "WG"}})
        ALL[label] = a
        ecrire_resume()

        # ---------- Partie 2 : pilotage par la direction de l'erreur ----------
        t4 = time.time()
        nd_ = min(N_DIR, n)
        moitie = np.arange(n) % 2                     # validation croisée en deux moitiés
        HC = Hs["C"][:, couche].astype(np.float32)
        HW = Hs["W"][:, couche].astype(np.float32)
        dirs = {}
        for h in (0, 1):
            tr = np.where(moitie != h)[0]
            u = HW[tr].mean(0) - HC[tr].mean(0)
            d = float(np.linalg.norm(u))
            X = np.concatenate([HC[tr], HW[tr]])
            st = StandardScaler().fit(X)
            lr = LogisticRegression(C=C_REG, max_iter=3000).fit(st.transform(X), np.r_[np.zeros(len(tr)), np.ones(len(tr))])
            wl = lr.coef_[0] / st.scale_
            dirs[h] = (u / d, d, wl / np.linalg.norm(wl))
        rngS = np.random.default_rng(SEED + 5)
        alea = rngS.normal(0, 1, (N_ALEA, HC.shape[1]))
        alea /= np.linalg.norm(alea, axis=1, keepdims=True)
        sel = list(range(nd_))
        dist = np.array([dirs[moitie[k]][1] for k in sel])
        norme_h = float(np.median(np.linalg.norm(HW[:nd_], axis=1)))

        def vecs_de(unite, alpha):
            """unite : fonction k → vecteur unitaire ; amplitude α × d (d estimée sur l'autre moitié)."""
            return np.stack([alpha * dist[m] * unite(k) for m, k in enumerate(sel)]).astype(np.float32)

        dir_moy = lambda k: dirs[moitie[k]][0]
        dir_log = lambda k: dirs[moitie[k]][2]
        sq = {r: [seqs[r][k] for k in sel] for r in "CW"}
        # runs W : autre = jeton de C (M monte si le modèle corrige) ; runs C : autre = jeton de W
        tk_a = {"W": [tokj["C"][k] for k in sel], "C": [tokj["W"][k] for k in sel]}
        tk_p = {"W": [tokj["W"][k] for k in sel], "C": [tokj["C"][k] for k in sel]}
        res = {}

        def mesurer(run, nom, couche_, vecs=None, capture=False):
            la, lp_, top, cap = passe_dirige(g, sq[run], tk_a[run], tk_p[run], couche_, vecs, capture)
            res[f"{run}:{nom}"] = {"M": la - lp_, "D": -lp_, "bascule": top == np.array(tk_a[run])}
            return cap

        t_ = time.time()
        capW = mesurer("W", "base", nL, None, True)
        est = (time.time() - t_) * (2 * N_ALEA + len(ALPHAS) + 8) / 60
        print(f"[PROGRESSION] {label} pilotage : premier passage en {time.time() - t_:.0f} s ; estimation ~{est:.0f} min", flush=True)
        capC = mesurer("C", "base", nL, None, True)
        mesurer("W", "crochet", nL, (capC - capW).astype(np.float32))
        mesurer("W", "remplace_L18", couche, (HC[:nd_] - HW[:nd_]).astype(np.float32))
        for al in ALPHAS + (-ALPHA_REF,):
            mesurer("W", f"erreur_{al:+d}", couche, vecs_de(dir_moy, al))
        mesurer("W", f"sonde_{ALPHA_REF:+d}", couche, vecs_de(dir_log, ALPHA_REF))
        mesurer("C", f"erreur_{ALPHA_REF:+d}", couche, vecs_de(dir_moy, ALPHA_REF))
        for r in range(N_ALEA):
            mesurer("W", f"alea{r}", couche, vecs_de(lambda k, r=r: alea[r], ALPHA_REF))
            mesurer("C", f"alea{r}", couche, vecs_de(lambda k, r=r: alea[r], ALPHA_REF))
        print(f"[MESURE] {label} pilotage ({len(res)} conditions × {nd_} problèmes) : {time.time() - t4:.0f} s", flush=True)

        def delta(run, nom, cle):
            return res[f"{run}:{nom}"][cle] - res[f"{run}:base"][cle]

        rngE = np.random.default_rng(SEED + 7)
        eff = {}
        for run in "WC":
            for cle in "MD":
                dr = np.stack([delta(run, f"alea{r}", cle) for r in range(N_ALEA)])
                eff[f"{run}:{cle}"] = ecart_alea(delta(run, f"erreur_{ALPHA_REF:+d}", cle), dr, rngE)
        crochet_dm = float(np.mean(delta("W", "crochet", "M")))
        verdict2 = regle_e2(eff["W:M"], eff["W:D"], crochet_dm, ok)
        desc = {nom.split(":", 1)[1]: {"dM": float(np.mean(delta("W", nom.split(":", 1)[1], "M"))),
                                       "dD": float(np.mean(delta("W", nom.split(":", 1)[1], "D"))),
                                       "bascule": float(np.mean(res[nom]["bascule"]))}
                for nom in res if nom.startswith("W:") and not nom.startswith("W:alea")}
        fe = lambda e: (f"{e['moy']:+.3f} (aléatoires {e['min_alea']:+.3f} à {e['max_alea']:+.3f}) ; écart {e['ecart']:+.3f} "
                        f"IC95 [{e['ci'][0]:+.3f} ; {e['ci'][1]:+.3f}]")
        print(f"\n[RÉSULTATS 2] {label} : {nd_} problèmes pilotés à L{couche}, α = {ALPHA_REF}, d médiane {np.median(dist):.2f}, "
              f"norme ajoutée / norme de l'état {ALPHA_REF * np.median(dist) / norme_h:.3f}")
        print(f"[BASE] {label} runs W : M médian {np.median(res['W:base']['M']):+.2f} nat, D médian {np.median(res['W:base']['D']):.2e}")
        print(f"[CROCHET] {label} remplacement de la dernière couche par celle du run C : ΔM {crochet_dm:+.2f} nat (seuil > {SEUIL_CROCHET})")
        print(f"[ΔM] {label} runs W, direction de l'erreur : {fe(eff['W:M'])}")
        print(f"[ΔD] {label} runs W, direction de l'erreur : {fe(eff['W:D'])}")
        print(f"[DOSE] {label} runs W : " + " ; ".join(f"{k} ΔM {v['dM']:+.3f} ΔD {v['dD']:+.3f} bascule {v['bascule']:.2f}"
                                                      for k, v in desc.items()))
        print(f"[RUNS C] {label} direction de l'erreur sur les runs C : ΔM {fe(eff['C:M'])} ; ΔD {fe(eff['C:D'])}")
        print(f"[VERDICT 2] {label} : {verdict2}")
        a.update({"verdict_pilotage": verdict2, "pilotage": {"n": nd_, "alpha": ALPHA_REF, "d_mediane": float(np.median(dist)),
                  "norme_etat_mediane": norme_h, "crochet_dM": crochet_dm, "effets": eff, "conditions_W": desc,
                  "base_M_mediane": float(np.median(res["W:base"]["M"]))}, "minutes": (time.time() - t_start) / 60})
        ALL[label] = a
        np.savez_compressed(os.path.join(OUT, f"{label}_b1e.npz"), idx=np.array(keep), jdiv=np.array(jdiv),
                            W=np.array(vals_["W"]), G=np.array(vals_["G"]),
                            **{f"score_{k}": v for k, v in sc0.items()},
                            **{f"{nom.replace(':', '_')}_{cle}": v[cle] for nom, v in res.items() for cle in ("M", "D")})
'''
s = s[:i1] + CORPS + s[i2:]

# ---- Synthèse
i3 = s.index("# ---------- Synthèse ----------")
s = s[:i3] + r'''# ---------- Synthèse ----------
print(f"\n{'=' * 100}\n[SYNTHÈSE] B1e — comparaison au-delà du dernier chiffre ; le modèle se sert-il du signal ?")
labs = [m for m in ALL if isinstance(ALL[m], dict) and "verdict" in ALL[m]]
v1 = {m: ALL[m]["verdict"] for m in labs}
v2 = {m: ALL[m].get("verdict_pilotage", "NON TESTABLE") for m in labs}
for m in labs:
    print(f"[SYNTHÈSE] {m:<22} : partie 1 {v1[m]} ; partie 2 {v2[m]}")
manque = len(labs) < len(MODELS)
these1 = "NON TESTABLE (modèle manquant)" if manque else synthese_e1(v1, ok)
these2 = "NON TESTABLE (modèle manquant)" if manque else synthese_e2(v2, ok)
ALL["these_comparaison"], ALL["these_pilotage"] = these1, these2
print(f"[SYNTHÈSE] thèse 1 « l'état caché compare la valeur écrite au produit vrai au-delà du dernier chiffre » : {these1} "
      f"(prédiction SOUTENUE 25 %, RÉFUTÉE 15 %)")
print(f"[SYNTHÈSE] thèse 2 « le modèle se sert de la direction de l'erreur pour corriger » : {these2} "
      f"(prédiction SOUTENUE 5 %, RÉFUTÉE 55 %)")
ecrire_resume()
print(f"[SORTIE] fichiers dans {OUT} : {sorted(os.listdir(OUT))}")
'''
open(dst, "w", encoding="utf-8").write(s)
print("ok", dst)
