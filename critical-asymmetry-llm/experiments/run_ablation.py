ABLATIONS = ["A1_meta_off", "A2_rl_off", "A3_memory_off", "A4_pid_off", "A5_T_only"]


def main():
    print("Ablations disponibles :")
    for a in ABLATIONS:
        print(f"  - {a}")


if __name__ == "__main__":
    main()
