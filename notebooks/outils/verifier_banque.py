"""Vérifie et assemble les banques de débuts de phrase produites par plusieurs modèles (prompt_banque_600.md).

Usage : python3 verifier_banque.py dossier/ [--sortie banque_finale.tsv]

Chaque fichier *.tsv ou *.txt du dossier est lu ligne par ligne ; seules les lignes à 9 champs séparés par des tabulations
sont prises en compte (les commentaires, blocs de code et lignes « ### SUITE » sont ignorés). Pour chaque ligne : contrôle
des champs, de la longueur, de la ponctuation finale, de la langue, du thème interdit et du genre attendu. Puis retrait des
doublons (texte identique après normalisation, ou presque identique : Jaccard des mots ≥ 0,8 dans la même langue), et
assemblage d'une banque finale de 100 thèmes complets (50 general, 50 technique ; 3 fr et 3 en par thème), en alternant
les fichiers. Écrit la banque, un rapport et l'empreinte md5 du texte de la banque (à dater avant toute exécution).
"""
import argparse
import collections
import hashlib
import os
import re
import sys
import unicodedata

CHAMPS = ["id", "theme_id", "registre", "domaine", "theme_fr", "theme_en", "langue", "genre", "texte"]
GENRES = ["EXP", "NAR", "HIS", "PRO", "REP", "IDR", "FOR", "ARG"]
DOM_G = ["nature", "histoire", "arts", "sport", "cuisine", "vie quotidienne", "géographie", "santé", "société",
         "technologie grand public"]
DOM_T = ["physique", "chimie", "biologie", "mathématiques", "informatique", "ingénierie", "médecine",
         "sciences de la Terre", "économie", "linguistique"]
# Thèmes interdits (déjà utilisés par A1c–A1z2). Expressions entières ; « * » final = préfixe.
EXPR = """photosynth* | révolution industrielle | industrial revolution | volcan* | jazz | cuisine italienne |
trou noir | trous noirs | black hole* | hockey | abeille* | bees | honeybee* | beehive* | intelligence artificielle |
artificial intelligence | saint-laurent | saint lawrence | renaissance | marée* | tides | vaccin* | grèce antique |
ancient greece | forêt boréale | forêts boréales | boreal forest* | théâtre | theater | theatre | électricité |
electricity | échecs | chess | chocolat | chocolate | pont suspendu | ponts suspendus | suspension bridge* |
mémoire humaine | human memory | comète* | comet* | recyclage | recycling | chromodynamique | chromodynamics |
catalyse enzymatique | enzyme catalysis | convolutif* | convolutional* | clé publique | public-key | public key |
hors équilibre | non-equilibrium | résonance magnétique | magnetic resonance | topologie algébrique |
algebraic topology | épigénét* | epigenetic* | supraconduct* | superconduct* | compilation à la volée |
just-in-time | turbulen* | bayésienne variationnelle | variational bayes* | tectonique | tectonic* |
ondes gravitationnelles | gravitational wave* | lithograph* | photolithograph* | théorie des catégories |
category theory | crispr | codes correcteurs | error-correcting | relativité générale | general relativity |
apprentissage par renforcement | reinforcement learning | empire romain | roman empire | tremblement* de terre |
séisme* | earthquake* | blues | cuisine japonaise | japanese cooking | japanese cuisine | étoile* à neutrons |
neutron star* | basket* | fourmi* | ants | ant colon* | apprentissage automatique | machine learning |
mississippi | courants océaniques | ocean current* | jardinage urbain | urban gardening | antibiotique* |
antibiotic* | égypte ancienne | ancient egypt | forêt tropicale | forêts tropicales | rainforest* | opéra | opera |
magnétisme | magnetism | poker | café | coffee | gratte-ciel | skyscraper* | sommeil | sleep | astéroïde* |
asteroid* | énergie solaire | solar power | solar energy""".replace("\n", " ").split("|")
EXPR = [e.strip() for e in EXPR if e.strip()]
GABARITS = ["voici ce qu'il faut savoir sur", "hier, un reportage consacré", "pour bien comprendre", "dans ce cours avancé sur",
            "une revue de littérature récente", "le formalisme mathématique de", "une erreur fréquente à propos",
            "here is what you need to know", "yesterday, a documentary about", "to really understand",
            "in this advanced lecture on", "a recent literature review on", "the mathematical formalism of",
            "a common misconception about"]
MOTS_FR = set("le la les des du de un une et est que qui dans pour sur au aux en par avec ce cette son sa ses il elle on "
              "pas plus se ne".split())
MOTS_EN = set("the a an and is are that which of in for on to with by this its it was were be from as at".split())


def norm(t):
    t = unicodedata.normalize("NFKC", t).lower().replace("’", "'")
    return re.sub(r"\s+", " ", re.sub(r"[^\w' -]", " ", t)).strip()


def genres_attendus(num):
    r = num % 8
    k = (r - 1) % 8                      # r = 1 → EXP en tête ; r = 0 → ARG en tête
    six = [GENRES[(k + i) % 8] for i in range(6)]
    return six[:3], six[3:]


def langue_probable(t):
    m = norm(t).split()
    fr, en = sum(w in MOTS_FR for w in m), sum(w in MOTS_EN for w in m)
    accents = bool(re.search(r"[éèêàùçôîâœ]", t))
    if fr + accents > en:
        return "fr"
    if en > fr:
        return "en"
    return "?"


def controler(ch):
    """Liste des défauts d'une ligne (9 champs)."""
    d = []
    _id, tid, reg, dom, tfr, ten, lg, ge, tx = [c.strip() for c in ch]
    m = re.fullmatch(r"T(\d{3})", tid)
    if not m or not 1 <= int(m.group(1)) <= 100:
        return ["theme_id invalide"]
    num = int(m.group(1))
    if reg != ("general" if num <= 50 else "technique"):
        d.append("registre")
    doms = DOM_G if num <= 50 else DOM_T
    if dom != doms[((num - 1) % 50) // 5]:
        d.append("domaine")
    if lg not in ("fr", "en"):
        d.append("langue")
    m2 = re.search(r"-(\d)$", _id)
    if not m2 or not 1 <= int(m2.group(1)) <= 6:
        d.append("id")
    else:
        pos = int(m2.group(1))
        gfr, gen = genres_attendus(num)
        if (pos <= 3 and lg != "fr") or (pos >= 4 and lg != "en"):
            d.append("langue/position")
        att = (gfr + gen)[pos - 1]
        if ge != att:
            d.append(f"genre (attendu {att})")
    n = len(tx.split())
    if not 6 <= n <= 16:
        d.append(f"longueur {n} mots")
    if re.search(r"[.?!…]\s*$", tx):
        d.append("ponctuation finale")
    if re.search(r"[\"«»“”()\t]", tx):
        d.append("caractère interdit")
    if lg in ("fr", "en") and langue_probable(tx) not in (lg, "?"):
        d.append("langue du texte")
    nt = norm(tfr + " " + ten)
    if any(re.search(r"(?<!\w)" + re.escape(e.rstrip("*")) + ("" if e.endswith("*") else r"(?!\w)"), nt) for e in EXPR):
        d.append("thème interdit")
    if any(norm(tx).startswith(norm(gb)) for gb in GABARITS):
        d.append("gabarit déjà utilisé")
    return d


def jaccard(a, b):
    a, b = set(a.split()), set(b.split())
    return len(a & b) / max(1, len(a | b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dossier")
    ap.add_argument("--sortie", default="banque_finale.tsv")
    args = ap.parse_args()
    fichiers = sorted(f for f in os.listdir(args.dossier) if f.endswith((".tsv", ".txt")))
    if not fichiers:
        sys.exit("aucun fichier .tsv ou .txt dans le dossier")
    rapport, themes = [], collections.OrderedDict()      # (fichier, theme_id) → lignes valides
    defauts = collections.Counter()
    for f in fichiers:
        ok = ko = 0
        for brut in open(os.path.join(args.dossier, f), encoding="utf-8-sig"):
            ch = brut.rstrip("\n\r").split("\t")
            if len(ch) != 9 or ch[0] == "id":
                continue
            d = controler(ch)
            if d:
                ko += 1
                defauts.update(x.split(" (")[0] for x in d)
                continue
            ok += 1
            themes.setdefault((f, ch[1].strip()), []).append([c.strip() for c in ch])
        rapport.append(f"{f} : {ok} lignes valides, {ko} rejetées")
    # doublons et quasi-doublons (ordre : fichiers alternés, thème par thème)
    par_f = collections.defaultdict(list)
    for (f, tid), lignes in themes.items():
        par_f[f].append((tid, lignes))
    ordre = []
    for k in range(max(len(v) for v in par_f.values())):
        for f in fichiers:
            if k < len(par_f[f]):
                ordre.append((f, *par_f[f][k]))
    vus = {"fr": [], "en": []}
    vus_themes, debut3 = set(), collections.Counter()
    finale = {"general": [], "technique": []}
    n_dup = n_theme_dup = n_incomplet = 0
    for f, tid, lignes in ordre:
        reg = lignes[0][2]
        cle_t = norm(lignes[0][5])
        if cle_t in vus_themes:
            n_theme_dup += 1
            continue
        garde = []
        for li in lignes:
            t = norm(li[8])
            if any(t == v or jaccard(t, v) >= 0.8 for v in vus[li[6]]):
                n_dup += 1
                continue
            garde.append(li)
        fr = [li for li in garde if li[6] == "fr"]
        en = [li for li in garde if li[6] == "en"]
        if len(fr) != 3 or len(en) != 3 or len(finale[reg]) >= 50:
            n_incomplet += len(fr) != 3 or len(en) != 3
            continue
        d3 = [(li[6], " ".join(norm(li[8]).split()[:3])) for li in fr + en]
        if any(debut3[x] >= 5 for x in d3):
            n_incomplet += 1
            continue
        debut3.update(d3)
        vus_themes.add(cle_t)
        for li in fr + en:
            vus[li[6]].append(norm(li[8]))
        finale[reg].append((f, tid, fr + en))
    total = sum(len(v) for v in finale.values())
    rapport += [f"défauts : {dict(defauts.most_common())}",
                f"thèmes en double entre fichiers écartés : {n_theme_dup} ; lignes en double ou quasi-double : {n_dup} ; "
                f"thèmes incomplets ou trop répétitifs écartés : {n_incomplet}",
                f"banque finale : {len(finale['general'])} thèmes general, {len(finale['technique'])} technique, "
                f"{6 * total} débuts de phrase"]
    with open(args.sortie, "w", encoding="utf-8") as out:
        out.write("\t".join(["nouvel_id", "source"] + CHAMPS) + "\n")
        k = 0
        for reg in ("general", "technique"):
            for f, tid, lignes in finale[reg]:
                k += 1
                for j, li in enumerate(lignes, 1):
                    out.write("\t".join([f"B{k:03d}-{j}", f] + li) + "\n")
    texte = "\n".join(li[8] for reg in ("general", "technique") for _, _, ls in finale[reg] for li in ls)
    md5 = hashlib.md5(texte.encode("utf-8")).hexdigest()
    rapport.append(f"empreinte md5 des textes : {md5} (à noter dans le suivi avant toute exécution)")
    if total < 100:
        rapport.append(f"INCOMPLET : il manque {100 - total} thèmes ; relancer un modèle (autre PRÉFIXE) et revérifier")
    print("\n".join(rapport))
    with open(os.path.splitext(args.sortie)[0] + "_rapport.txt", "w", encoding="utf-8") as fr_:
        fr_.write("\n".join(rapport) + "\n")


if __name__ == "__main__":
    main()
