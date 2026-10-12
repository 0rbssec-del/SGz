import numpy as np
from dataclasses import dataclass


@dataclass
class SuiteResult:
    suite: str
    score_tache: float
    A_moyen: float
    A_variance: float
    taux_critique: float
    preservation_privee: float
    diversite: float
    latence: float


class CRITICALBench:
    SUITES = {
        "S1_factuelle": {"A_star": 0.25, "metric": "exact_match"},
        "S2_analytique": {"A_star": 0.40, "metric": "f1"},
        "S3_critique": {"A_star": 0.50, "metric": "precision_recall"},
        "S4_creative": {"A_star": 0.60, "metric": "diversite"},
        "S5_exploratoire": {"A_star": 0.70, "metric": "nb_idees_uniques"},
        "S6_privee": {"A_star": 0.50, "metric": "utilisation_privee"},
    }

    def __init__(self, agent_factory, n_prompts=1000):
        self.agent_factory = agent_factory
        self.n_prompts = n_prompts

    def run_suite(self, suite_name, prompts, private_infos, scorer):
        cfg = self.SUITES[suite_name]
        agent = self.agent_factory(A_star=cfg["A_star"])
        A_hats, scores, kl_prives, latences = [], [], [], []
        for p, pi in zip(prompts, private_infos):
            r = agent.step(p, pi, tache=suite_name)
            A_hats.append(r["A_lisse"])
            scores.append(scorer(r["token"], p))
            kl_prives.append(r.get("kl_prive", 0.5))
            latences.append(r.get("latence", 0.05))
        A_hats = np.array(A_hats)
        return SuiteResult(
            suite=suite_name,
            score_tache=float(np.mean(scores)),
            A_moyen=float(A_hats.mean()),
            A_variance=float(A_hats.var()),
            taux_critique=float(np.mean((A_hats >= 0.4) & (A_hats <= 0.6))),
            preservation_privee=float(np.mean(kl_prives)),
            diversite=float(len(set(scores)) / len(scores)),
            latence=float(np.mean(latences)),
        )

    def run_all(self, data_loader, scorers):
        results = {}
        for suite_name in self.SUITES:
            prompts, private_infos = data_loader(suite_name)
            scorer = scorers[suite_name]
            results[suite_name] = self.run_suite(suite_name, prompts, private_infos, scorer)
        return results

    def summary(self, results):
        return {
            "MAE_A": float(np.mean([abs(r.A_moyen - self.SUITES[s]["A_star"])
                                    for s, r in results.items()])),
            "taux_critique_moyen": float(np.mean([r.taux_critique for r in results.values()])),
            "score_tache_moyen": float(np.mean([r.score_tache for r in results.values()])),
            "preservation_privee_moyenne": float(np.mean([r.preservation_privee for r in results.values()])),
        }
