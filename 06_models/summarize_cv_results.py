from shared.summary import summarize_results


def main():
    summary = summarize_results()
    available = sorted(set(summary["available"]))
    print(f"available experiments: {available}")
    print(f"missing experiments: {summary['missing']}")
    if len(summary["comparison"]):
        print("\nExperiment comparison")
        print(summary["comparison"].to_string(index=False))
        print("\nsaved 06_models/results/summary/experiment_comparison.csv")
    if len(summary["fold_comparison"]):
        print("saved 06_models/results/summary/fold_comparison.csv")


if __name__ == "__main__":
    main()
