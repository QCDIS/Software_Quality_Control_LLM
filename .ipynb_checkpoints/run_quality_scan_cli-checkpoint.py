import argparse # For parsing command-line arguments
import json # For outputting raw results as JSON
from evaluation.evaluator import evaluate_metrics
from lifecycle.stage_manager import get_metrics_for_stage

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Run notebook quality scan from the command line.")
    parser.add_argument("--stage", type=str, required=True, help="Lifecycle stage (e.g., Development, Maintenance)")
    parser.add_argument("--path", type=str, required=True, help="Path to notebook file or project folder")
    parser.add_argument("--github", type=str, default=None, help="GitHub repo URL (optional, only needed for FAIRness checks)")
    parser.add_argument("--save", type=str, help="Optional: path to save raw results as JSON")

    # Parse arguments
    args = parser.parse_args()

    print(f"Running quality scan...")
    print(f"Stage: {args.stage}")
    print(f"Path: {args.path}")
    if args.github:
        print(f"GitHub URL: {args.github}")

    # Step 1: Get metrics for selected stage
    metrics = get_metrics_for_stage(args.stage)

    # Step 2: Run the tool on the target path
    results = evaluate_metrics(metrics, path=args.path, github_url=args.github)

    # Step 3: Print the raw output exactly (as JSON)
    print("\n Raw Restuls: \n")
    print(json.dumps(results, indent=2, ensure_ascii=False))

    # Step 4: Optionally save to JSON file
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n Results saved to: {args.save}")

if __name__ == "__main__":
    main()