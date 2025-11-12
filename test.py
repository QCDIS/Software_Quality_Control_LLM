#### LLM THINGS ####
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pathlib import Path
from openai import OpenAI
from collections.abc import Mapping, Sequence
import numbers
from transformers import StoppingCriteria, StoppingCriteriaList




class StopOnSequences(StoppingCriteria):
    def __init__(self, stop_ids, window=50):
        super().__init__()
        self.stop_ids = [torch.tensor(s, dtype=torch.long) for s in stop_ids]
        self.window = window

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Only scan the last window tokens for speed
        seq = input_ids[0][-self.window:]
        for s in self.stop_ids:
            if len(seq) >= len(s) and torch.equal(seq[-len(s):], s.to(seq.device)):
                return True
        return False

MODEL_CANDIDATES = [
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "deepseek-ai/deepseek-coder-1.3b-instruct",
    "bigcode/starcoder2-3b",
    "meta-llama/CodeLlama-7b-Instruct-hf",
]




prompt = """


import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
import seaborn as sns  # For statistical data visualization
from sklearn.ensemble import RandomForestClassifier  # For the Random Forest model
from sklearn.metrics import classification_report, accuracy_score  # For model evaluation
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets

try:
    glass_data = pd.read_csv("glass_classification.csv")
except FileNotFoundError:
    raise FileNotFoundError("The dataset 'glass_classification.csv' was not found. Please ensure it is in the correct directory.")

print(glass_data.head())

print(glass_data.tail())

print("Dataset shape:", glass_data.shape)

print(glass_data.info())
print(glass_data.describe())
print("Missing values in each column:\n", glass_data.isnull().sum())


sns.catplot(x='Type of glass', data=glass_data, kind='count')
plt.title('Count of Glass Types')
plt.show()

features = ['Al', 'RI', 'Si', 'Na', 'Ca']
for feature in features:
    plt.figure(figsize=(4, 4))
    sns.barplot(x='Type of glass', data=glass_data, y=feature)
    plt.title(f'{feature} by Type of Glass')
    plt.show()

correlation = glass_data.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(correlation, cbar=True, fmt='.1f', annot=True, annot_kws={'size': 8}, cmap="YlOrBr")
plt.title('Correlation Heatmap')
plt.show()

X = glass_data.drop('Type of glass', axis=1)  # Features
Y = glass_data['Type of glass']  # Target variable

# Display features and target variable
print("Features:\n", X.head())
print("Target variable:\n", Y.head())

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# Display the shapes of the training and testing sets
print("Training set shape:", X_train.shape, Y_train.shape)
print("Testing set shape:", X_test.shape, Y_test.shape)

model = RandomForestClassifier(random_state=42)  # Initialize the model with a random state for reproducibility
model.fit(X_train, Y_train)  # Fit the model to the training data

X_train_prediction = model.predict(X_train)
train_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Training Accuracy:', train_accuracy)

X_test_prediction = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Testing Accuracy:', test_accuracy)

print("Classification Report for Training Data:\n", classification_report(Y_train, X_train_prediction))
print("Classification Report for Testing Data:\n", classification_report(Y_test, X_test_prediction))

input_data = [203, 1.51514, 14.85, 0, 2.42, 73.72, 0, 8.39, 0.56, 0]  # Example input
numpy_data = np.asarray(input_data)  # Convert input to numpy array
reshaped_input = numpy_data.reshape(1, -1)  # Reshape for prediction

predict = model.predict(reshaped_input)
glass_types = {
    1: "building_windows_float_processed",
    2: "building_windows_non_float_processed",
    3: "vehicle_windows_float_processed",
    4: "vehicle_windows_non_float_processed",
    5: "containers",
    6: "tableware",
    7: "headlamps"
}
 

PUT CODE INSIDE A MULTIPLE FUNCTIONS FOR MAINTAINABILITY (**Modularize into small functions**)

ADD DETAILED COMMENTS (**Explain logic and intent**)
ONLY OYTPUT THE FUNCTION TO SORT AND NOTHING ELSE 
AFTER COMPLETING THE CODE GENERATE A TOKEN </END_OF_CODE>
"""

#print(prompt)
# need to test with different models with a loop
MODEL_ID = MODEL_CANDIDATES[0]

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)

# Build pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
)
enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
input_len = enc["input_ids"].shape[-1]
max_ctx = getattr(model.config, "max_position_embeddings", None)
if max_ctx is None or max_ctx == float("inf"):
    max_ctx = tokenizer.model_max_length  # sometimes absurdly large; still fine if smaller than real

# Keep a small safety margin for special tokens
SAFETY_MARGIN = 16
available = int(max_ctx - input_len - SAFETY_MARGIN)
max_new = max(1, available)  # at least 1


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

stop_text = "</END_OF_CODE>"
stop_ids = [tokenizer.encode(stop_text, add_special_tokens=False)]
stops = StoppingCriteriaList([StopOnSequences(stop_ids, window=64)])

outputs = pipe(
    prompt,  # make your prompt instruct the model to end with ``` after the code
    max_new_tokens=1024,                 # avoid very large values
    temperature=0.7,                     # a bit higher temp helps avoid loops
    top_p=0.9,
    top_k=50,
    do_sample=True,
    no_repeat_ngram_size=6,              # block repeated phrases
    repetition_penalty=1.18,             # discourage verbatim loops
    eos_token_id=tokenizer.eos_token_id, # still keep EOS
    pad_token_id=tokenizer.eos_token_id,
    stopping_criteria=stops,             # stop at ```
    return_full_text=False,
)

print(outputs[0]["generated_text"])