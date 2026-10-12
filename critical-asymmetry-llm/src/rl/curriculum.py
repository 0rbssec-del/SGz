PHASES = [
    {"name": "S1_factuelle", "A_star": 0.25, "epochs": 100},
    {"name": "S2_analytique", "A_star": 0.40, "epochs": 100},
    {"name": "S3_critique", "A_star": 0.50, "epochs": 200},
    {"name": "S4_creative", "A_star": 0.60, "epochs": 150},
    {"name": "S5_exploratoire", "A_star": 0.70, "epochs": 150},
    {"name": "S6_privee", "A_star": 0.50, "epochs": 200},
]


def get_phase(epoch):
    cumulative = 0
    for phase in PHASES:
        cumulative += phase["epochs"]
        if epoch < cumulative:
            return phase
    return PHASES[-1]
