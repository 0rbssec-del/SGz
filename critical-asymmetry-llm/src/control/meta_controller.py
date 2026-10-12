from collections import deque
import numpy as np


class MetaController:
    MODES = ["stabilisateur", "excitateur", "freinateur", "explorateur"]

    def __init__(self, llm_m2=None, A_min=0.2, A_max=0.8, history=16):
        self.M2 = llm_m2
        self.A_min, self.A_max = A_min, A_max
        self.historique = deque(maxlen=history)
        self.A_star = 0.5

    def gradient(self, A_hat):
        if len(self.historique) < 2:
            return 0.0
        return A_hat - self.historique[-2]

    def detect_mode(self, A_hat):
        grad = self.gradient(A_hat)
        if abs(grad) > 0.05:
            return "stabilisateur"
        if A_hat < 0.35:
            return "excitateur"
        if A_hat > 0.70:
            return "freinateur"
        return "explorateur"

    def politique(self, mode, A_hat):
        if mode == "stabilisateur":
            return -0.2 * self.gradient(A_hat)
        if mode == "excitateur":
            return +0.1
        if mode == "freinateur":
            return -0.1
        return 0.05 * np.sin(0.1 * len(self.historique))

    def decide(self, tache, contexte, A_hat, enjeu=0.5):
        mode = self.detect_mode(A_hat)
        delta = self.politique(mode, A_hat)
        A_star_llm = self.A_star
        if self.M2 is not None:
            prompt = (f"Tache: {tache}\nContexte: {contexte[:500]}\n"
                      f"Asymetrie actuelle: {A_hat:.2f}\n"
                      f"Historique: {list(self.historique)}\n"
                      f"Enjeu: {enjeu}\nMode: {mode}\n"
                      "Quelle cible A* dans [0.2, 0.8] ? Reponds uniquement par un nombre.")
            try:
                A_star_llm = float(self.M2.generate(prompt).strip().split()[0])
            except Exception:
                A_star_llm = self.A_star
        A_star = 0.5 * (self.A_star + delta) + 0.5 * A_star_llm
        A_star = float(np.clip(A_star, self.A_min, self.A_max))
        self.historique.append(A_star)
        self.A_star = float(np.mean(self.historique))
        return self.A_star
