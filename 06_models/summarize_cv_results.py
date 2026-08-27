import argparse

from shared.summary import summarize_all_existing, summarize_report_tables


def parse_args():
    ap = argparse.ArgumentParser(description="Summarize CV results.")
    ap.add_argument("--all-existing", action="store_true",
                    help="include diagnostic and historical result folders with status labels")
    return ap.parse_args()


def main():
    args = parse_args()
    summary = summarize_all_existing() if args.all_existing else summarize_report_tables()
    out_label = "06_models/results/archive/historical_result_inventory" if args.all_existing else "06_models/results/summary"
    available = sorted(set(summary["available"]))
    print(f"available experiments: {available}")
    print(f"missing experiments: {summary['missing']}")
    if len(summary["comparison"]):
        print("\nExperiment comparison")
        print(summary["comparison"].to_string(index=False))
        filename = "all_experiments_classified.csv" if args.all_existing else "main_model_comparison.csv"
        print(f"\nsaved {out_label}/{filename}")
    if len(summary["fold_comparison"]):
        filename = "all_fold_results_classified.csv" if args.all_existing else "main_fold_comparison.csv"
        print(f"saved {out_label}/{filename}")
    if not args.all_existing:
        if len(summary.get("supplementary_full_cv", [])):
            print("saved 06_models/results/summary/supplementary_full_cv.csv")
        if len(summary.get("iberia_diagnostics", [])):
            print("saved 06_models/results/summary/iberia_diagnostics.csv")


if __name__ == "__main__":
    main()
