from pathlib import Path
import ast
import statistics

def run_modularity_check(target_path):
    """
    Project-level modularity analysis using file structure and function count heuristics.

    Args:
        target_path (str): root directory of the Python project

    Returns:
        dict: structured output with detailed diagnostic information
    """
    py_files = []
    for file_path in Path(target_path).rglob("*.py"):
        if "venv" in file_path.parts or "__pycache__" in file_path.parts:
            continue
        py_files.append(file_path)

    if not py_files:
        return {
            "status": "warn",
            "message": "⚠️ No Python files found in the target directory.",
        }

    func_counts = {}
    for file in py_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                node = ast.parse(f.read())
                func_counts[file.name] = sum(isinstance(n, ast.FunctionDef) for n in ast.walk(node))
        except Exception:
            continue

    num_files = len(func_counts)
    total_funcs = sum(func_counts.values())
    avg_funcs_per_file = total_funcs / num_files if num_files else 0
    std_dev_funcs = statistics.stdev(func_counts.values()) if len(func_counts) > 1 else 0

    # Build interpretation
    observations = []
    if num_files < 5:
        observations.append("• The project contains very few Python files. This may indicate insufficient modularization.")
    if avg_funcs_per_file < 2:
        observations.append("• On average, each file has fewer than 2 functions. Consider increasing functional abstraction.")
    if std_dev_funcs > 3:
        observations.append("• Function distribution varies significantly between files. Try balancing responsibilities.")

    return {
        "status": "warn",
        "message": (
            "⚠️ <b>Modularity Assessment</b>:<br>"
            "This evaluation reviews the structure of modules and their functional composition.<br><br>"
            "<u>Project Summary:</u><br>"
            f"- Total Python files analyzed: <b>{num_files}</b><br>"
            f"- Total function definitions: <b>{total_funcs}</b><br>"
            f"- Average functions per file: <b>{avg_funcs_per_file:.2f}</b><br>"
            f"- Std. deviation of functions/file: <b>{std_dev_funcs:.2f}</b><br><br>"
            "<u>Insights:</u><br>"
            + "<br>".join(observations) if observations else
            "The project shows a reasonably modular structure based on file/function balance."
        )
    }
