from benchmarks.critical_bench import CRITICALBench


def dummy_data_loader(suite_name):
    prompts = [f"{suite_name}_prompt_{i}" for i in range(100)]
    privates = [f"private_{i}" for i in range(100)]
    return prompts, privates


def main():
    bench = CRITICALBench(agent_factory=lambda A_star: None, n_prompts=100)
    print("CRITICAL-Bench pret.")
    print(f"Suites : {list(bench.SUITES.keys())}")


if __name__ == "__main__":
    main()
