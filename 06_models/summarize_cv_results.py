import argparse

from shared.summary import summarize_all_existing, summarize_results


def parse_args():
    ap = argparse.ArgumentParser(description="Summarize CV results.")
    ap.add_argument("--all-existing", action="store_true",
                    help="include diagnostic and historical result folders with status labels")
    return ap.parse_args()


def main():
    args = parse_args()
    summary = summarize_all_existing() if args.all_existing else summarize_results()
    out_label = "06_models/results/summary/all_existing" if args.all_existing else "06_models/results/summary"
    available = sorted(set(summary["available"]))
    print(f"available experiments: {available}")
    print(f"missing experiments: {summary['missing']}")
    if len(summary["comparison"]):
        print("\nExperiment comparison")
        print(summary["comparison"].to_string(index=False))
        print(f"\nsaved {out_label}/experiment_comparison.csv")
    if len(summary["fold_comparison"]):
        print(f"saved {out_label}/fold_comparison.csv")


if __name__ == "__main__":
    main()
