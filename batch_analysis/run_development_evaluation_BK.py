import os
import json
import subprocess
from bs4 import BeautifulSoup # For stripping HTML
from pathlib import Path 

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pathlib import Path
from openai import OpenAI




# --- Configuration ---
base_dir = "/home/nafis/Development/jupyter-quality-extension/11_envri_validation_set"  
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


def save_code(path: str, code: str) -> None:
    Path(path).write_text(code, encoding="utf-8")

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


# Go through all projects in base_dir
for project in os.listdir(base_dir):
    project_path = os.path.join(base_dir, project)
    if not os.path.isdir(project_path):
        continue

    print(f"Running analysis for: {project}")

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
    except subprocess.CalledProcessError as e:
        print(f"Error running CLI for {project}: {e}")
        continue

    stdout = result.stdout.strip()
    if not stdout:
        print(f"⚠ No output for {project}")
        continue

    # Find the json block from the CLI output
    json_start = stdout.find("{")
    if json_start == -1:
        print(f"⚠ Failed to locate JSON result in CLI output for {project}")
        continue

    try:
        parsed_results = json.loads(stdout[json_start:])
    except json.JSONDecodeError as e:
        print(f"⚠ JSON decode error for {project}: {e}")
        continue

    # Clean up the parsed_results in-place 
    for section_name, section_metrics in parsed_results.items():
        for metric_name, metric_result in section_metrics.items():
            if isinstance(metric_result, dict) and "message" in metric_result:
                message = metric_result["message"]

                # Case 1: Code Smells - keep only pylint score line 
                # if metric_name == "Code Smells":
                    
                #     # lines = message.splitlines()

                #     if isinstance(message, str):
                #         lines = message.splitlines()
                #     else:
                #         lines = str(message).splitlines()

                #     score_line = next(
                #         (line.strip() for line in lines if line.strip().startswith("Your code has been rated at")),
                #         None 
                #     )
                #     if score_line:
                #         metric_result["message"] = score_line
                #     else:
                #         metric_result["message"] = "N/A"
                #     continue


                # Case 1: Code Smells - keep only pylint score line 
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

    # Store cleaned result into summary
    summary_results[project] = parsed_results


    # keys = list(parsed_results.keys())
    # print(keys)
    # print(parsed_results[keys[1]])
    # print("---------------------------")
    # print("printing projec path ", project_path)
    # looping should be here ...
    counter = 5
    while True:
        # Give code + error message to llm
        keys = list(parsed_results.keys())
        #print("printing the keys ", keys)
        
        general_error_message = parsed_results[keys[0]]
        
        for i in range(1, len(keys)):
            code_file_path = keys[i]
            special_error_message = parsed_results[keys[i]]

            code_str = Path(code_file_path).read_text(encoding="utf-8")
            print(code_file_path)
            print(general_error_message)
            print(special_error_message)
            print("#######################################")
            prompt = f"""
                Given Code:
                {code_str}

                General Code Issue:
                {general_error_message}

                Special Code Issue: 
                {special_error_message}

                You are an expert in software engineering. Based on the given code and general code issues and special code issues, 
                generate a better version of the code such that the error messages are improved.

                ADD APPROPRIATE COMMENTS
                PUT CODE INSIDE A FUNCTION

                Only generate code.
            """

            #print(prompt)
            # ned to test with different models with a loop
            # MODEL_ID = MODEL_CANDIDATES[0]

            # tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
            # model = AutoModelForCausalLM.from_pretrained(
            #     MODEL_ID,
            #     torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            #     device_map="auto",
            # )

            # # Build pipeline
            # pipe = pipeline(
            #     "text-generation",
            #     model=model,
            #     tokenizer=tokenizer,
            #     device_map="auto",
            # )
            # enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
            # input_len = enc["input_ids"].shape[-1]
            # max_ctx = getattr(model.config, "max_position_embeddings", None)
            # if max_ctx is None or max_ctx == float("inf"):
            #     max_ctx = tokenizer.model_max_length  # sometimes absurdly large; still fine if smaller than real

            # # Keep a small safety margin for special tokens
            # SAFETY_MARGIN = 16
            # available = int(max_ctx - input_len - SAFETY_MARGIN)
            # max_new = max(1, available)  # at least 1


            # outputs = pipe(
            #     prompt,
            #     max_new_tokens=4096,
            #     temperature=0.2,
            #     top_p=0.9,
            #     do_sample=True,
            #     repetition_penalty=1.05,
            #     eos_token_id=tokenizer.eos_token_id,
            #     pad_token_id=tokenizer.eos_token_id,
            #     return_full_text=False,
            # )

            MODEL = "gpt-4o-mini"  # small, fast, cost-efficient model
            client = OpenAI()

            def generate_code_from_prompt(prompt: str) -> str:
                resp = client.responses.create(
                    model=MODEL,
                    input=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    top_p=0.95,
                    # You can omit max_output_tokens to let the API choose,
                    # or set it explicitly if you want a ceiling (e.g., 4000)
                    # max_output_tokens=4000,
                )
                return resp.output_text  # convenience accessor for the text

            #generated_text = outputs[0]["generated_text"]
            generated_text = generate_code_from_prompt(prompt)
            #print("### GENERATING CODE ####")
            #print(generated_text)
            save_code(code_file_path, generated_text)


        counter -= 1

        #break
       
       
        




# --- Save final results ---
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(summary_results, f, indent=2)

print(f"\n ✓ All done! Results saved to: {output_file}")


