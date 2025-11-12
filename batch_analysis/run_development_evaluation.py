import os
import json
import subprocess
from bs4 import BeautifulSoup # For stripping HTML
from pathlib import Path 

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pathlib import Path
from openai import OpenAI
from collections.abc import Mapping, Sequence
import numbers
import re
from collections import defaultdict
from statistics import fmean




# --- Configuration ---
base_dir = "/home/nafis/Development/Software_Quality_Control_LLM/11_envri_validation_set"  
selected_stage = "Development"  

parent_dir = os.path.dirname(base_dir) # Get parent directory of base_dir
results_dir = os.path.join(parent_dir, "11_envri_validation_set_results") # Define results directory path
os.makedirs(results_dir, exist_ok=True)  # Make sure the folder exists
output_file = os.path.join(results_dir, "batch_development_results_test.json") # Define output file path inside the results directory

summary_results = {}  # Final dictionary to hold all project outputs

#### LLM THINGS ####

MODEL_CANDIDATES = [
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "deepseek-ai/deepseek-coder-1.3b-instruct",
    "bigcode/starcoder2-3b",
]

from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


def append_if_number(store, key, value):
    if isinstance(value, numbers.Number) and not isinstance(value, bool):
        store[key].append(value)


def save_code(path: str, code: str) -> None:
    Path(path).write_text(code, encoding="utf-8")

R_RATIO = r'([0-9]+(?:\.[0-9]+)?)\s*/\s*([0-9]+(?:\.[0-9]+)?)'
R_MSG = re.compile(
    rf"rated at\s+{R_RATIO}\s*\(previous run:\s*{R_RATIO}\s*,\s*([+-]?[0-9]+(?:\.[0-9]+)?)\)",
    flags=re.IGNORECASE
)

def iter_numeric_kv_with_code_smells(obj, prefix="", parents=()):
    """
    Yield (path, value) for:
      - all numeric leaves (ints/floats, excluding bools)
      - 'Code Smells.status' (string)
      - parsed numbers from 'Code Smells.message':
          - rating_num, rating_den, rating (ratio)
          - previous_num, previous_den, previous (ratio)
          - delta
    """
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            new_prefix = f"{prefix}.{k}" if prefix else str(k)
            yield from iter_numeric_kv_with_code_smells(v, new_prefix, parents + (k,))
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        for i, v in enumerate(obj):
            new_prefix = f"{prefix}[{i}]"
            yield from iter_numeric_kv_with_code_smells(v, new_prefix, parents + (f"[{i}]",))
    else:
        # 1) Always include numeric leaves
        if isinstance(obj, numbers.Number) and not isinstance(obj, bool):
            yield prefix, obj
            return

        # 2) Include 'Code Smells.status' as a string
        if len(parents) >= 2 and parents[-2] == "Code Smells" and parents[-1] == "status":
            yield prefix, obj  # e.g., "Code Smells.status" -> "fail"

        # 3) Parse numbers from 'Code Smells.message'
        if len(parents) >= 2 and parents[-2] == "Code Smells" and parents[-1] == "message" and isinstance(obj, str):
            m = R_MSG.search(obj)
            if m:
                rating_num = float(m.group(1))
                rating_den = float(m.group(2))
                prev_num   = float(m.group(3))
                prev_den   = float(m.group(4))
                delta      = float(m.group(5))

                # Emit both numerator/denominator and computed ratios
                yield f"{prefix}.rating_num", rating_num
                yield f"{prefix}.rating_den", rating_den
                if rating_den != 0:
                    yield f"{prefix}.rating", rating_num / rating_den

                yield f"{prefix}.previous_num", prev_num
                yield f"{prefix}.previous_den", prev_den
                if prev_den != 0:
                    yield f"{prefix}.previous", prev_num / prev_den

                yield f"{prefix}.delta", delta
            else:
                # Optional: fall back to generic numeric extraction from the string
                for n, d in re.findall(R_RATIO, obj):
                    n, d = float(n), float(d)
                    yield f"{prefix}.ratio_num", n
                    yield f"{prefix}.ratio_den", d
                    if d != 0:
                        yield f"{prefix}.ratio", n / d


def iter_numeric_kv(obj, prefix=""):
    """
    Yield (path, numeric_value) for every leaf that is numeric.
    Treats bools as non-numeric.
    """
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            new_prefix = f"{prefix}.{k}" if prefix else str(k)
            yield from iter_numeric_kv(v, new_prefix)
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        for i, v in enumerate(obj):
            new_prefix = f"{prefix}[{i}]"
            yield from iter_numeric_kv(v, new_prefix)
    else:
        if isinstance(obj, numbers.Number) and not isinstance(obj, bool):
            yield prefix, obj


def load_hf_pipeline():
    last_err = None
    for name in MODEL_CANDIDATES:
        try:
            tok = AutoTokenizer.from_pretrained(name, use_fast=True)
            mdl = AutoModelForCausalLM.from_pretrained(
                name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
            return pipeline(
                "text-generation",
                model=mdl,
                tokenizer=tok,
                device_map="auto",
            )
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not load any model: {last_err}")


def run_cli(project_path):
    # Run the CLI tool and capture its stdout
    try:
        result = subprocess.run(
            [
                "python",
                "run_quality_scan_cli.py",
                "--stage", selected_stage,
                "--path", project_path
            ],
            capture_output=True,
            text=True,
            check=True
        )
        stdout = result.stdout.strip()
        #print("stdout", stdout)
        if not stdout:
            print(f"⚠ No output for {project}")
            return False
        return stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running CLI for {project}: {e}")
        

def parse_json(stdout):
    # Find the json block from the CLI output
    json_start = stdout.find("{")
    if json_start == -1:
        print(f"⚠ Failed to locate JSON result in CLI output for {project}")
        

    try:
        parsed_results = json.loads(stdout[json_start:])
    except json.JSONDecodeError as e:
        print(f"⚠ JSON decode error for {project}: {e}")
        return False
    
    return parsed_results
    

def clean_parsed_results(parsed_results):
    for section_name, section_metrics in parsed_results.items():
        for metric_name, metric_result in section_metrics.items():
            if isinstance(metric_result, dict) and "message" in metric_result:
                message = metric_result["message"]
                if metric_name == "Code Smells":
                    try:
                        soup = BeautifulSoup(str(message), "html.parser")
                        divs = soup.find_all("div")

                        score_line = None 
                        for div in divs:
                            text = div.get_text(strip=True)
                            if text.startswith("Your code has been rated at"):
                                score_line = text
                                break

                        metric_result["message"] = score_line if score_line else "N/A"
                    
                    except Exception as e:
                        metric_result["message"] = f"N/A (parse error: {e})"

                    continue
                    

                # Case 2: Other messages - clean HTML and keep top 1-2 lines only
                clean_text = BeautifulSoup(message, "html.parser").get_text(separator="\n")
                lines = [line.strip() for line in clean_text.splitlines() if line.strip()]

                # Filter out lines that start with 'Tip:' or 'Note:'
                main_lines = [line for line in lines if not line.lower().startswith(("tip:", "note:"))]

                # Keep only the first 1-2 relevant lines
                trimmed = main_lines[:2] if main_lines else ["N/A"]
                metric_result["message"] = " ".join(trimmed)
    return parsed_results


# Go through all projects in base_dir
global_results = defaultdict(list)
for project in os.listdir(base_dir):
    
    project_path = os.path.join(base_dir, project)
    if not os.path.isdir(project_path):
        continue

    print(f"Running analysis for: {project}")

    # Run the CLI tool and capture its stdout
    stdout = run_cli(project_path)
    if not stdout:
        continue

    # Find the json block from the CLI output
    parsed_results = parse_json(stdout)
    if not parsed_results: 
        continue


    # Clean up the parsed_results in-place 
    parsed_results = clean_parsed_results(parsed_results)    

    # Store cleaned result into summary
    summary_results[project] = parsed_results
    
    keys = list(parsed_results.keys())    
    
    # Iterating over each file
    local_results = defaultdict(list)
    for i in range(1, len(keys)):

        code_file_path = keys[i]
        general_error_message = parsed_results[keys[0]]
        special_error_message = parsed_results[keys[i]]
        code_str = Path(code_file_path).read_text(encoding="utf-8")
        
        # FORMATTING
        general_error_message = dict(iter_numeric_kv_with_code_smells(general_error_message)) 
        special_error_message = dict(iter_numeric_kv_with_code_smells(special_error_message)) 

        print("################     INITIAL   #######################")
        print(code_file_path)
        print(general_error_message)
        print(special_error_message)
    
        print("#######################################")
        counter = 3
        while counter > 0:
            print("Inside Loop")
            # gpt-5-nano
            # gpt-4o-mini
            MODEL = "gpt-5-nano"  # small, fast, cost-efficient model
            client = OpenAI()

          

            prompt = f"""
                You are an expert in software engineering. Based on the given code and general code issues and special code issues, 
                generate a better version of the code such that the error messages are improved.

                ADD DETAILED COMMENTS (**Explain logic and intent**) -> Higher score is better
                PUT CODE INSIDE A MULTIPLE FUNCTIONS FOR MAINTAINABILITY (**Modularize into small functions**) Higher score is better
                MAKE SURE THERE ARE NO DUPLICATE CODE (**Eliminate repetition with abstractions**) Lower score is better
                MAKE SURE THE CODE IS MAINTANABLE (**Readable, modular, well-tested design**) Higher score is better
                LOW CYCLOMATIC COMPLEXITY (**Target medium complexity range**) Medium score (6-10 or 11-20) is better
                
                Given Code:
                {code_str}

                General Code Issue:
                {general_error_message}

                Special Code Issue: 
                {special_error_message}

                Only generate code. 
                DO NOT GENERATE ANY QUOTATION LIKE ```
            """

            def generate_code_from_prompt(prompt: str) -> str:
                resp = client.responses.create(
                    model=MODEL,
                    input=[{"role": "user", "content": prompt}],
                    # TEMPERATURE AND TOP_P NOT SUPPORTED FOR GPT 5
                    #temperature=0.2,
                    #top_p=0.95,
                    # You can omit max_output_tokens to let the API choose,
                    # or set it explicitly if you want a ceiling (e.g., 4000)
                    # max_output_tokens=4000,
                )
                return resp.output_text  # convenience accessor for the text

            #generated_text = outputs[0]["generated_text"]
            code_str = generate_code_from_prompt(prompt)
            save_code(code_file_path, code_str)
            # print("Printing code")
            # print(code_str)
            print("Printing project path", project_path)
            local_stdout = run_cli(project_path)
            if not local_stdout:
                continue
            #print("Local STDOUT ",local_stdout)
            
            local_parsed_results = parse_json(local_stdout)
            if not local_parsed_results: 
                continue

            local_parsed_results = clean_parsed_results(local_parsed_results)   

            local_keys = list(local_parsed_results.keys())    
            local_general_error_message = local_parsed_results[keys[0]]
            local_special_error_message = local_parsed_results[keys[i]]

            general_error_message = local_general_error_message
            special_error_message = local_special_error_message

            # FORMATTING
            # iter_numeric_kv_with_code_smells
            # iter_numeric_kv
            general_error_message = dict(iter_numeric_kv_with_code_smells(general_error_message)) 
            special_error_message = dict(iter_numeric_kv_with_code_smells(special_error_message)) 

            print(general_error_message)
            print(special_error_message)
            print(counter, "iteration done")


            summary_results[project] = local_parsed_results

            #print("### GENERATING CODE ####")
            #print(generated_text)
            counter -= 1
            



        #break

        # loop ends
        # Do final calculation
        print("General Error Message and Special Error Message")
        """
        Printing project path /home/nafis/Development/jupyter-quality-extension/11_envri_validation_set/Glassid-Classification-
        {'Software Size (LoC).loc': 870, 'Code Duplication.percentage': 0.0, 'Code Duplication.duplicated_lines': 0, 'Code Duplication.total_lines': 1154}
        {'Code Smells.status': 'fail', 'Code Smells.message.rating_num': 8.5, 'Code Smells.message.rating_den': 10.0, 'Code Smells.message.rating': 0.85, 'Code Smells.message.previous_num': 8.53, 'Code Smells.message.previous_den': 10.0, 'Code Smells.message.previous': 0.853, 'Code Smells.message.delta': -0.04, 
        'Maintainability Index.score': 39.58524957755014, 'Cyclomatic Complexity.score': 2.75, 'Comment Density.density': 12.637362637362637}
        """

        # Average per project
        LoC = general_error_message.get("Software Size (LoC).loc")
        code_duplication_percentage = general_error_message.get("Code Duplication.percentage")
        code_duplication_duplicated_lines = general_error_message.get("Code Duplication.duplicated_lines")
        code_duplication_total_lines = general_error_message.get("Code Duplication.total_lines")

        # From: {'Code Smells.status': 'fail', 'Code Smells.message.rating_num': 8.5, ...}
        code_smells_status = special_error_message.get("Code Smells.status")
        code_smells_rating_num = special_error_message.get("Code Smells.message.rating_num")
        code_smells_rating_den = special_error_message.get("Code Smells.message.rating_den")
        code_smells_rating = special_error_message.get("Code Smells.message.rating")  # e.g., 0.85
        code_smells_previous_num = special_error_message.get("Code Smells.message.previous_num")
        code_smells_previous_den = special_error_message.get("Code Smells.message.previous_den")
        code_smells_previous = special_error_message.get("Code Smells.message.previous")
        code_smells_delta = special_error_message.get("Code Smells.message.delta")

        maintainability_index_score = special_error_message.get("Maintainability Index.score")
        cyclomatic_complexity_score = special_error_message.get("Cyclomatic Complexity.score")
        comment_density_density = special_error_message.get("Comment Density.density")

        # print(f"LoC = {LoC}")
        # print(f"code_duplication_percentage = {code_duplication_percentage}")
        # print(f"code_duplication_duplicated_lines = {code_duplication_duplicated_lines}")
        # print(f"code_duplication_total_lines = {code_duplication_total_lines}")

        # print(f"code_smells_status = {code_smells_status}")
        # print(f"code_smells_rating_num = {code_smells_rating_num}")
        # print(f"code_smells_rating_den = {code_smells_rating_den}")
        # print(f"code_smells_rating = {code_smells_rating}")
        # print(f"code_smells_previous_num = {code_smells_previous_num}")
        # print(f"code_smells_previous_den = {code_smells_previous_den}")
        # print(f"code_smells_previous = {code_smells_previous}")
        # print(f"code_smells_delta = {code_smells_delta}")

        # print(f"maintainability_index_score = {maintainability_index_score}")
        # print(f"cyclomatic_complexity_score = {cyclomatic_complexity_score}")
        # print(f"comment_density_density = {comment_density_density}")
        print("You are appending .........")

        append_if_number(local_results, "Software Size (LoC).loc", LoC)
        append_if_number(local_results, "Code Duplication.percentage", code_duplication_percentage)
        append_if_number(local_results, "Code Duplication.duplicated_lines", code_duplication_duplicated_lines)
        append_if_number(local_results, "Code Duplication.total_lines", code_duplication_total_lines)

        append_if_number(local_results, "Code Smells.message.rating_num", code_smells_rating_num)
        append_if_number(local_results, "Code Smells.message.rating_den", code_smells_rating_den)
        append_if_number(local_results, "Code Smells.message.rating", code_smells_rating)
        append_if_number(local_results, "Code Smells.message.previous_num", code_smells_previous_num)
        append_if_number(local_results, "Code Smells.message.previous_den", code_smells_previous_den)
        append_if_number(local_results, "Code Smells.message.previous", code_smells_previous)
        append_if_number(local_results, "Code Smells.message.delta", code_smells_delta)

        append_if_number(local_results, "Maintainability Index.score", maintainability_index_score)
        append_if_number(local_results, "Cyclomatic Complexity.score", cyclomatic_complexity_score)
        append_if_number(local_results, "Comment Density.density", comment_density_density)


    # Average of all projects
    print("Printing local results ", local_results)
    print(local_results)
    averages = {key: fmean(vals) for key, vals in local_results.items() if vals}
    print("printing averages ", averages)
    for key, avg in averages.items():
        global_results.setdefault(key, []).append(avg)

    print("Global result ...", global_results)

       
        


global_averages = {key: fmean(vals) for key, vals in global_results.items() if vals}
print("FINALLY ")
print("Global averge ", global_averages)
# --- Save final results ---
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(summary_results, f, indent=2)

print(f"\n ✓ All done! Results saved to: {output_file}")


