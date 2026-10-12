import numpy as np
import torch
from ..asymmetry.multimodal_metrics import (
    phi7_intermodal_kl, phi8_visual_salience, phi9_private_visual_kl,
    compute_A_multimodal,
)
from ..control.pid import PIDController, SlidingWindowA
from ..control.multimodal_controller import MultiModalController
from ..control.meta_controller import MetaController
from ..memory.private_memory import PrivateMemory


class MultiModalCAAAgent:
    def __init__(self, encoder, agent_id=0, A_star=0.5, device="cpu"):
        self.id = agent_id
        self.encoder = encoder.to(device) if encoder is not None else None
        self.device = device
        self.A_star = A_star
        self.pid = PIDController(A_star=A_star)
        self.window = SlidingWindowA(window=16)
        self.controller = MultiModalController()
        self.meta = MetaController()
        self.memory = PrivateMemory().to(device)

    @torch.no_grad()
    def step(self, input_ids, attention_mask, pixel_values,
             pixel_values_private, tache="general"):
        if self.encoder is None:
            import random
            A_hat = random.uniform(0.4, 0.6)
            return {"A_hat": A_hat, "A_lisse": A_hat, "A_star": self.A_star,
                    "T": 1.0, "p": 0.9, "lambda_v": 0.5, "tau_v": 1.0,
                    "phis": {"A_MM": A_hat}, "surprise": 0.5}
        out_pub = self.encoder(input_ids, attention_mask, pixel_values,
                               lambda_v=self.controller.lambda_v)
        out_priv = self.encoder(input_ids, attention_mask, pixel_values_private,
                                lambda_v=self.controller.lambda_v)
        logits_sans_image = self.encoder.text(input_ids=input_ids,
                                              attention_mask=attention_mask).last_hidden_state.mean(dim=1)
        logits_avec_image = out_pub["text"].mean(dim=1)
        logits_priv = out_priv["text"].mean(dim=1)
        phi7 = phi7_intermodal_kl(logits_sans_image, logits_avec_image)
        phi8 = phi8_visual_salience(out_pub["attn_vt"])
        phi9 = phi9_private_visual_kl(logits_avec_image, logits_priv)
        phi1 = phi2 = phi3 = phi4 = phi5 = phi6 = 0.5
        A_dict = compute_A_multimodal(phi1, phi2, phi3, phi4, phi5, phi6,
                                      phi7, phi8, phi9)
        A_hat = A_dict["A_MM"]
        A_lisse = self.window.update(A_hat)
        self.A_star = self.meta.decide(tache, "", A_lisse)
        delta_u = self.pid.step(A_lisse)
        T, p, lambda_v, tau_v = self.controller.update(delta_u)
        self.memory.update(-np.log(0.5))
        return {"A_hat": A_hat, "A_lisse": A_lisse, "A_star": self.A_star,
                "T": T, "p": p, "lambda_v": lambda_v, "tau_v": tau_v,
                "phis": A_dict, "surprise": -np.log(0.5)}
