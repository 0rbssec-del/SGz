import numpy as np
from ..asymmetry.metrics import phi2_kl_libre_aligne, phi6_private_asymmetry
from ..control.pid import PIDController, SlidingWindowA
from ..control.sampler import TemperatureTopPController
from ..control.meta_controller import MetaController
from ..memory.private_memory import PrivateMemory


class CAAAgent:
    def __init__(self, model, agent_id=0, A_star=0.5, device="cpu"):
        self.id = agent_id
        self.model = model
        self.device = device
        self.pid = PIDController(A_star=A_star)
        self.window = SlidingWindowA(window=16)
        self.sampler = TemperatureTopPController()
        self.meta = MetaController()
        self.memory = PrivateMemory().to(device)
        self.A_star = A_star

    def step(self, prompt, private_info, tache="general"):
        context = f"{private_info}\n\n{prompt}"
        if self.model is None:
            import random
            A_hat = random.uniform(0.4, 0.6)
            return {"token": 0, "A_hat": A_hat, "A_lisse": A_hat,
                    "A_star": self.A_star, "T": 1.0, "p": 0.9, "surprise": 0.5}
        logits_libre = self.model.logits(context)
        logits_aligne = self.model.logits(context, system="aligned")
        phi2 = phi2_kl_libre_aligne(logits_libre, logits_aligne)
        phi6 = phi6_private_asymmetry(self.model, [prompt], private_info, self.device)
        A_hat = (phi2 + phi6) / 2
        A_lisse = self.window.update(A_hat)
        self.A_star = self.meta.decide(tache, prompt, A_lisse)
        delta_u = self.pid.step(A_lisse)
        T, p = self.sampler.update(delta_u)
        token = self.model.sample(logits_libre, T=T, p=p)
        return {"token": token, "A_hat": A_hat, "A_lisse": A_lisse,
                "A_star": self.A_star, "T": T, "p": p, "surprise": -np.log(0.5)}

    def feedback(self, surprise, neighbor_a=None):
        self.memory.update(surprise, neighbor_a)
