# %%
!pip install dotenv unsloth accelerate

# %%
from unsloth import FastLanguageModel

import os
from tqdm import tqdm
import accelerate
from dotenv import load_dotenv
from huggingface_hub import login

import torch
from torch.nn.functional import cosine_similarity
from transformers import pipeline
import numpy as np

# %% [markdown]
# # Model Loadings

# %%
ALL_LABELS = [
    'Text Production', 'Visual Formatting', 'Clarity', 'Section Planning',
    'Structural', 'Object Insertion', 'Cross-reference', 'Fluency',
    'Idea Generation', 'Idea Organization', 'Citation Integration', 'Coherence',
    'Linguistic Style', 'Scientific Accuracy', 'Macro Insertion'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
load_dotenv()
login(os.getenv("OPENAI_API"))

writing_model_name = "minnesotanlp/scholawrite-llama3.1-8b-writing"
writing_model, writing_tokenizer = FastLanguageModel.from_pretrained(
    model_name=writing_model_name,
    max_seq_length=4096,
    load_in_4bit=True,
    dtype=None,
)
FastLanguageModel.for_inference(writing_model)

# %%
pred_model_name = "minnesotanlp/scholawrite-llama3.1-8b-writing"
pred_model, pred_tokenizer = FastLanguageModel.from_pretrained(
    model_name=pred_model_name,
    max_seq_length=4096,
    load_in_4bit=True,
    dtype=None,
)
FastLanguageModel.for_inference(pred_model)

# %% [markdown]
# # Model Inferences

# %%
persona_definition = {
  "Idea Generation": "formulate and record initial thoughts and concepts.",
  "Idea Organization": "select the most useful materials and demarcate those generated ideas in a visually formatted way.",
  "Section Planning": "initially create sections and sub-level structures.",
  "Text Production": "translate your ideas into full languages, either from your language or borrowed sentences from an external source.",
  "Object Insertion": "insert visual claims of your arguments (e.g., figures, tables, equations, footnotes, itemized lists, etc.).",
  "Cross-reference": "link different sections, figures, tables, or other elements within a document through referencing commands.",
  "Citation Integration": "incorporate bibliographic references into a document and systematically link these references using citation commands.",
  "Macro Insertion": "incorporate predefined commands orc packages into a LaTeX document to alter its formatting.",
  "Fluency": "fix grammatical or syntactic errors in the text or LaTeX commands.",
  "Coherence": "logically link (1) any of the two or multiple sentences within the same paragraph; (2) any two subsequent paragraphs; or (3) objects to be consistent as a whole.",
  "Structural": "improve the flow of information by modifying the location of texts and objects.",
  "Clarity": "improve the semantic relationships between texts to be more straightforward and concise.",
  "Linguistic Style": "modify texts with your writing preferences regarding styles and word choices, etc.",
  "Scientific Accuracy": "update or correct scientific evidence (e.g., numbers, equations) for more accurate claims.",
  "Visual Formatting": "modify the stylistic formatting of texts, objects, and citations."
}

def class_prompt(before_text):
    usr_prompt= f"""Here are all the possible writing intention labels:

Idea Generation: Formulate and record initial thoughts and concepts.
Idea Organization: Select the most useful materials and demarcate those generated ideas in a visually formatted way.
Section Planning: Initially create sections and sub-level structures.
Text Production: Translate their ideas into full languages, either from the writers' language or borrowed sentences from an external source.
Object Insertion: Insert visual claims of their arguments (e.g., figures, tables, equations, footnotes, itemized lists, etc.).
Cross-reference: Link different sections, figures, tables, or other elements within a document through referencing commands.
Citation Integration: Incorporate bibliographic references into a document and systematically link these references using citation commands.
Macro Insertion: Incorporate predefined commands or packages into a LaTeX document to alter its formatting.
Fluency: Fix grammatical or syntactic errors in the text or LaTeX commands.
Coherence: Logically link (1) any of the two or multiple sentences within the same paragraph; (2) any two subsequent paragraphs; or (3) objects to be consistent as a whole.
Structural: Improve the flow of information by modifying the location of texts and objects.
Clarity: Improve the semantic relationships between texts to be more straightforward and concise.
Linguistic Style: Modify texts with the writer's writing preferences regarding styles and word choices, etc.
Scientific Accuracy: Update or correct scientific evidence (e.g., numbers, equations) for more accurate claims.
Visual Formatting: Modify the stylistic formatting of texts, objects, and citations.

Identify the most likely next writing intention of a graduate researcher when writing the following LaTex paper draft. Your output should only be a label from the list above.

{before_text}"""
  
    return [
        {"role": "user", "content": usr_prompt}
    ]


def text_gen_prompt(before_text, writing_intention):

    user_prompt = f"""You are a computer science researcher with extensive experience in scholarly writing. Here, you are writing a research paper in natural language processing using LaTeX.

You currently want to {persona_definition[writing_intention]}

Below is the paper you have written so far. Given the paper information below and the corresponding scholarly writing intention, please revise or add to the text to fulfill this writing intention.

You may insert, delete, or revise text at appropriate places in the given paper.

Please provide a complete output. Do not generate text that is nonsensical or unrelated to the given paper information.

Your response should limited to 2000 word tokens

{before_text}"""

    return [
        {"role": "user", "content": user_prompt}
    ]

# %%
def process_label(predicted_label):
    all_labels = ['Text Production', 'Visual Formatting', 'Clarity', 'Section Planning',
 'Structural', 'Object Insertion', 'Cross-reference', 'Fluency',
 'Idea Generation', 'Idea Organization', 'Citation Integration', 'Coherence',
 'Linguistic Style', 'Scientific Accuracy', 'Macro Insertion']

 
    if predicted_label not in all_labels:
        found = 0
        for true_label in all_labels:
            if true_label in predicted_label:
                predicted_label = true_label
                found = 1
                break
        
        # If the output from gpt didn't contain any expeceted label
        if found != 1:
            predicted_label = "Text Production"
    
    return predicted_label

# %%
def predict_intention(text, model, tokenizer):
  text = class_prompt(text)
  input_ids = tokenizer.apply_chat_template(text, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)

  outputs = model.generate(input_ids, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)

  response = tokenizer.batch_decode(outputs)

  response = response[0].split("<|start_header_id|>assistant<|end_header_id|>")[1].strip()

  predicted_class= process_label(response)

  return predicted_class

# %%
def writing_inference(before_text, writing_intention, model, tokenizer):
  text = text_gen_prompt(before_text, writing_intention)
  input_ids = tokenizer.apply_chat_template(text, max_length=4096, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)

  outputs = model.generate(input_ids, max_new_tokens=len(before_text)+100, do_sample=True, top_k=50, top_p=0.95)

  response = tokenizer.batch_decode(outputs)

  response = response[0].split("<|start_header_id|>assistant<|end_header_id|>")[1].strip()
  response = response.replace("<|eot_id|>", "")

  return response

# %%
def clean_text(text):
    text = re.sub(r"<del>.*?<\/del>", "", text, flags=re.DOTALL)

    text = re.sub(r"<same>(.*?)<\/same>", r"\1", text, flags=re.DOTALL)

    text = re.sub(r"<add>(.*?)<\/add>", r"\1", text, flags=re.DOTALL)

    tags_to_remove = ["<add>", "</add>", "<del>", "</del>", "<same>", "</same>"]
    for tag in tags_to_remove:
        text = text.replace(tag, "")
    
    return text

def save_raw_output(output, writing_intention, i):
  generation_raw_dir = '/kaggle/working/'
  with open(f"{generation_raw_dir}/iter_generation_{i}.txt", "w") as file:
    file.write(output)

  with open(f"{intention_raw_dir}/iter_intention_{i}.txt", "w") as file:
    file.write(writing_intention)

# %%
import re

# %%
import shutil

# %%
import glob
import os

pattern = "/kaggle/working/iter_*"
files = glob.glob(pattern)

for file_path in files:
    if os.path.isfile(file_path):
        os.remove(file_path)
        print(f"Deleted {file_path}")


# %%
def setup(seed_name):
  global generation_dir, intention_dir, generation_raw_dir, intention_raw_dir, path_to_seed

  output_dir = f"/kaggle/working/{seed_name}"

  generation_dir = f"{output_dir}/generation"
  intention_dir = f"{output_dir}/intention"
  generation_raw_dir = f"{output_dir}/generation_raw"
  intention_raw_dir = f"{output_dir}/intention_raw"

  os.makedirs(generation_dir, exist_ok=True)
  os.makedirs(intention_dir, exist_ok=True)
  os.makedirs(generation_raw_dir, exist_ok=True)
  os.makedirs(intention_raw_dir, exist_ok=True)

  path_to_seed = f"/kaggle/input/scholawrite-seeds/{seed_name}.txt"

def load_seed(fname):
  with open(fname, 'r') as file:
    return file.read()

# Step 0: Truncate prev_writing to avoid exceeding token limit
def truncate_text_to_max_tokens(text, tokenizer, max_tokens=2048, reserved_tokens=512):
    tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    if len(tokens) > (max_tokens - reserved_tokens):
        truncated_tokens = tokens[-(max_tokens - reserved_tokens):] 
        text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    return text

def aggregate_iterative_writing(pred_model, pred_tokenizer, writing_model, writing_tokenizer, i=0):
  if i==0:
    prev_writing = load_seed(path_to_seed)
  else:
    # load the i-1 iteration generation file for previous writing 
    prev_gen_path = os.path.join(generation_dir, f"iter_generation_{i-1}.txt")
    with open(prev_gen_path, "r", encoding="utf-8") as f:
      prev_writing = f.read()

  pbar = tqdm(total=100)

  with torch.no_grad():
    while i < 100:
      prev_writing = truncate_text_to_max_tokens(prev_writing, pred_tokenizer)

      # Step 1: Predict intention based on previous version
      print("----------Iteration ", i, " started-----------")
      writing_intention = predict_intention(prev_writing, pred_model, pred_tokenizer)
      print("iteration ", i, ": ", writing_intention)

      # Step 2: Generate new version based on predicted intention
      output = writing_inference(prev_writing, writing_intention, writing_model, writing_tokenizer)
      print("iteration ", i, ": finished writing inference")
      
      # Step 3: Clean generated output
      cleaned_output = clean_text(output)

      # Step 4: Save intermediate outputs
      with open(f"{generation_dir}/iter_generation_{i}.txt", "w") as f_gen:
        f_gen.write(cleaned_output)
        print("saved ", f"{generation_dir}/iter_generation_{i}.txt")

      with open(f"{intention_dir}/iter_intention_{i}.txt", "w") as f_intent:
        f_intent.write(writing_intention)
        print("saved ", f"{intention_dir}/iter_generation_{i}.txt")

      save_raw_output(output, writing_intention, i)  # saves raw output with tags

      # Step 5: Update state for next iteration
      prev_writing = cleaned_output
      i += 1
      pbar.update(1)


def main(pred_model, pred_tokenizer, writing_model, writing_tokenizer, i=0):
  for seed_name in ["seed1"]:
    print(f"Working on {seed_name}")
    setup(seed_name)
    aggregate_iterative_writing(pred_model, pred_tokenizer, writing_model, writing_tokenizer, i)
    print("-" * 100)

# %% [markdown]
# ## Iterative Writing with Data Reference
# ### Data Processing

# %%
import pandas as pd

data1 = pd.read_parquet('/kaggle/input/scholawrite/all_sorted-00000-of-00002.parquet', engine='pyarrow')
data2 = pd.read_parquet('/kaggle/input/scholawrite/all_sorted-00001-of-00002.parquet', engine='pyarrow')
data1.shape, data2.shape

# %%
df = pd.concat([data1, data2], axis=0, ignore_index=True)

# %%
import difflib

def char_level_diff(before, after):
    matcher = difflib.SequenceMatcher(None, before, after)
    del_text_parts = []
    add_text_parts = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace' or tag == 'delete':
            del_text_parts.append(before[i1:i2])
        if tag == 'replace' or tag == 'insert':
            add_text_parts.append(after[j1:j2])

    return ''.join(del_text_parts).strip(), ''.join(add_text_parts).strip()

df['del_text'], df['add_text'] = zip(*df.apply(lambda row: char_level_diff(row['before text'], row['after text']), axis=1))

# %%
def paragraph_diff(before_text, after_text):
    before_paras = [p.strip() for p in before_text.split('\n') if p.strip()]
    after_paras = [p.strip() for p in after_text.split('\n') if p.strip()]

    matcher = difflib.SequenceMatcher(None, before_paras, after_paras)
    
    changed_before = []
    changed_after = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':
            changed_before.extend(before_paras[i1:i2])
            changed_after.extend(after_paras[j1:j2])

    return '\n'.join(changed_before).strip(), '\n'.join(changed_after).strip()

# %%
df['before_paragraph'], df['after_paragraph'] = zip(*df.apply(
    lambda row: paragraph_diff(row['before text'], row['after text']),
    axis=1
))

# %%
def balanced_sample(df, n_per_label):
    sampled = (
        df[['label', 'before_paragraph', 'after_paragraph']].groupby('label')
        .apply(lambda g: g.sample(n=min(n_per_label, len(g)), random_state=42))
        .reset_index(drop=True)
    )
    return sampled

# %%
df['before_len'] = df['before_paragraph'].str.len()
df['after_len'] = df['after_paragraph'].str.len()

print("Before Paragraph Length Stats:")
print(df['before_len'].describe())

print("\nAfter Paragraph Length Stats:")
print(df['after_len'].describe())

# %%
n_per_label = 20
df_filtered = df[
    (df['before_len'] > 50) & (df['after_len'] > 50) &
    (df['before_len'] < 800) & (df['after_len'] < 800)
]

final_prompt_df = balanced_sample(df_filtered, n_per_label)

print(final_prompt_df['label'].value_counts())

# %% [markdown]
# ### Inference Stage

# %%
def class_prompt_with_data_ref(before_text, examples):
    example_block = "\n\n".join([
        f"""### Example
Before:
{ex['before_paragraph']}

After:
{ex['after_paragraph']}

Label:
{ex['label']}""" for ex in examples
    ])

    usr_prompt = f"""Here are all the possible writing intention labels:

{chr(10).join([f"{label}: {desc}" for label, desc in persona_definition.items()])}

Below are examples of revisions and their corresponding writing intentions:

{example_block}

Now, identify the most likely next writing intention of a graduate researcher when writing the following LaTex paper draft. Your output should only be a label from the list above.

{before_text}"""

    return [{"role": "user", "content": usr_prompt}]

# %%
def text_gen_prompt_with_data_ref(before_text, writing_intention, examples):
    example_block = "\n\n".join([
        f"""### Example
Writing Intention: {ex['label']}

Before:
{ex['before_paragraph']}

After:
{ex['after_paragraph']}""" for ex in examples
    ])

    user_prompt = f"""

Do not copy the structure of the original text.
Avoid repeating section templates or formatting unless explicitly necessary.
Focus on content transformation, not literal repetition.

You are a computer science researcher with extensive experience in scholarly writing. Here, you are writing a research paper in natural language processing using LaTeX.

You currently want to {persona_definition[writing_intention]}

Below are examples of how other researchers revised their drafts to fulfill various scholarly writing intentions:

{example_block}

Below is the paper you have written so far. Given the paper information below and the corresponding scholarly writing intention, please revise or add to the text to fulfill this writing intention.

You may insert, delete, or revise text at appropriate places in the given paper.

Please provide a complete output. Do not generate text that is nonsensical or unrelated to the given paper information.

Your response should be limited to 2000 word tokens.

{before_text}"""

    return [{"role": "user", "content": user_prompt}]


# %%
examples = final_prompt_df[['before_paragraph', 'after_paragraph', 'label']].sample(n=10, random_state=42).to_dict(orient='records')

# %%
def predict_intention_with_data_ref(text, model, tokenizer):
  text = class_prompt_with_data_ref(text, examples)
  input_ids = tokenizer.apply_chat_template(text, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)

  outputs = model.generate(input_ids, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)

  response = tokenizer.batch_decode(outputs)

  response = response[0].split("<|start_header_id|>assistant<|end_header_id|>")[1].strip()

  predicted_class= process_label(response)

  return predicted_class

# %%
def writing_inference_with_data_ref(before_text, writing_intention, model, tokenizer):
  text = text_gen_prompt_with_data_ref(before_text, writing_intention, examples)
  input_ids = tokenizer.apply_chat_template(text, max_length=4096, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)

  outputs = model.generate(input_ids, max_new_tokens=len(before_text)+100, do_sample=True, top_k=50, top_p=0.95)

  response = tokenizer.batch_decode(outputs)

  response = response[0].split("<|start_header_id|>assistant<|end_header_id|>")[1].strip()
  response = response.replace("<|eot_id|>", "")

  return response

# %%
def aggregate_iterative_writing_with_data_ref(pred_model, pred_tokenizer, writing_model, writing_tokenizer, i=0):
  if i==0:
    prev_writing = load_seed(path_to_seed)
  else:
    # load the i-1 iteration generation file for previous writing 
    prev_gen_path = os.path.join(generation_dir, f"iter_generation_{i-1}.txt")
    with open(prev_gen_path, "r", encoding="utf-8") as f:
      prev_writing = f.read()
  pbar = tqdm(total=100)

  with torch.no_grad():
    while i < 100:
      prev_writing = truncate_text_to_max_tokens(prev_writing, pred_tokenizer)

      # Step 1: Predict intention based on previous version
      print("----------Iteration ", i, " started-----------")
      writing_intention = predict_intention_with_data_ref(prev_writing, pred_model, pred_tokenizer)
      print("iteration ", i, ": ", writing_intention)

      # Step 2: Generate new version based on predicted intention
      output = writing_inference_with_data_ref(prev_writing, writing_intention, writing_model, writing_tokenizer)
      print("iteration ", i, ": finished writing inference")
      
      # Step 3: Clean generated output
      cleaned_output = clean_text(output)

      # Step 4: Save intermediate outputs
      with open(f"{generation_dir}/iter_generation_{i}.txt", "w") as f_gen:
        f_gen.write(cleaned_output)
        print("saved ", f"{generation_dir}/iter_generation_{i}.txt")

      with open(f"{intention_dir}/iter_intention_{i}.txt", "w") as f_intent:
        f_intent.write(writing_intention)
        print("saved ", f"{intention_dir}/iter_generation_{i}.txt")

      save_raw_output(output, writing_intention, i)  # saves raw output with tags

      # Step 5: Update state for next iteration
      prev_writing = cleaned_output
      i += 1
      pbar.update(1)


def main_with_data_ref(pred_model, pred_tokenizer, writing_model, writing_tokenizer, i=0):
  for seed_name in ["seed1"]:
    print(f"Working on {seed_name}")
    setup(seed_name)
    aggregate_iterative_writing_with_data_ref(pred_model, pred_tokenizer, writing_model, writing_tokenizer, i)
    print("-" * 100)

# %%
main_with_data_ref(pred_model, pred_tokenizer, writing_model, writing_tokenizer)
import shutil
shutil.make_archive("/kaggle/working/all_llama-sw_with_data_ref_outputs", 'zip', "/kaggle/working/seed1")

# %%


# %% [markdown]
# ## Iterative Writing

# %%
main(pred_model, pred_tokenizer, writing_model, writing_tokenizer)

# %%
shutil.make_archive("/kaggle/working/all_llama_sw_outputs", 'zip', "/kaggle/working/seed1")

# %%
# remove previous zip files save storages
os.remove("/kaggle/working/all_llama_sw_outputs.zip")
# os.remove("/kaggle/working/all_outputs.zip")

# %% [markdown]
# ## Early Analysis for Output(Seed 1)

# %%
import pandas as pd
import difflib

def tokenize(text):
    return text.split()

def compare_texts(before_text, after_text):
    before_tokens = tokenize(before_text)
    after_tokens = tokenize(after_text)
    
    diff = list(difflib.ndiff(before_tokens, after_tokens))
    
    added = [word[2:] for word in diff if word.startswith('+ ')]
    deleted = [word[2:] for word in diff if word.startswith('- ')]

    return added, deleted

def process_iterations(base_path):
    records = []

    for i in range(99):  
        file1 = os.path.join(base_path, f"iter_generation_{i}.txt")
        file2 = os.path.join(base_path, f"iter_generation_{i+1}.txt")
        
        if not os.path.exists(file1) or not os.path.exists(file2):
            continue
        
        with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
            text1 = f1.read()
            text2 = f2.read()
        
        added, deleted = compare_texts(text1, text2)
        
        records.append({
            'iteration': i,
            'add_words': added,
            'del_words': deleted
        })

    df = pd.DataFrame(records)
    return df


base_path = "/kaggle/working/seed1/generation"

df_diffs = process_iterations(base_path)

print(df_diffs.head())

# %%
def load_intentions(base_path, total_iterations=100):
    intentions = []
    for i in range(total_iterations):
        file_path = os.path.join(base_path, f"iter_intention_{i}.txt")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                intentions.append(f.read().strip())
        else:
            intentions.append(None) 
    return intentions

intention_path = "/kaggle/working/seed1/intention"
intention_list = load_intentions(intention_path)
df_diffs['intention'] = intention_list[:99]

print(df_diffs.head())

# %%
df_diffs.to_csv("/kaggle/working/iteration_word_diffs.csv", index=False)

# %% [markdown]
# ## Clear Storage

# %%
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# %%
import gc

del pred_model
del pred_tokenizer
del writing_model
del writing_tokenizer

gc.collect()  
torch.cuda.empty_cache()

# %% [markdown]
# ## Meta-LLaMa

# %%
def load_classifier_model():
  model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

  model, tokenizer = FastLanguageModel.from_pretrained(
  model_name=model_name,
  max_seq_length=4096,
  load_in_4bit=True,
  dtype=None
  )

  FastLanguageModel.for_inference(model)

  return model, tokenizer

# %%
meta_model, meta_tokenizer = load_classifier_model()

# %%
main_with_data_ref(meta_model, meta_tokenizer, meta_model, meta_tokenizer)
import shutil
shutil.make_archive("/kaggle/working/all_llama-meta_with_data_ref_outputs", 'zip', "/kaggle/working/seed1")

# %%
main(meta_model, meta_tokenizer, meta_model, meta_tokenizer)
shutil.make_archive("/kaggle/working/all_llama_meta_outputs", 'zip', "/kaggle/working/seed1")

# %%
import shutil
shutil.make_archive("/kaggle/working/all_llama_meta_outputs", 'zip', "/kaggle/working/seed1")

# %%
import shutil

shutil.make_archive("/kaggle/working/all_outputs", 'zip', "/kaggle/working")

# %% [markdown]
# ## GPT4o - 2nd experiment

# %%
from openai import OpenAI
import os
import re
from tqdm import tqdm

client = OpenAI(api_key="OPENAI_API")
OPENAI_MODEL = "gpt-4o"

def predict_intention_openai(text):
    messages = class_prompt(text)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.5,
        max_tokens=100,
    )
    raw_label = response.choices[0].message.content.strip()
    return process_label(raw_label)


def writing_inference_openai(before_text, writing_intention):
    messages = text_gen_prompt(before_text, writing_intention)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7,
        max_tokens=1500,
    )
    return response.choices[0].message.content.strip()


def aggregate_iterative_writing_openai():
    prev_writing = load_seed(path_to_seed)
    i = 0
    pbar = tqdm(total=100)

    while i < 100:
        # Step 1: Predict next intention
        writing_intention = predict_intention_openai(prev_writing)

        # Step 2: Generate revised text
        output = writing_inference_openai(prev_writing, writing_intention)

        # Step 3: Clean output
        cleaned_output = clean_text(output)

        # Step 4: Save outputs
        with open(f"{generation_dir}/iter_generation_{i}.txt", "w") as f_gen:
            f_gen.write(cleaned_output)

        with open(f"{intention_dir}/iter_intention_{i}.txt", "w") as f_intent:
            f_intent.write(writing_intention)

        save_raw_output(output, writing_intention, i)
        print("finished iteration ", i, writing_intention)

        # Step 5: Update state
        prev_writing = cleaned_output
        i += 1
        pbar.update(1)


# %%
def setup_openai(seed_name):
    global generation_dir, intention_dir, generation_raw_dir, intention_raw_dir, path_to_seed

    output_dir = f"/kaggle/working/{seed_name}_openai"

    generation_dir = f"{output_dir}/generation_openai"
    intention_dir = f"{output_dir}/intention_openai"
    generation_raw_dir = f"{output_dir}/generation_raw_openai"
    intention_raw_dir = f"{output_dir}/intention_raw_openai"

    os.makedirs(generation_dir, exist_ok=True)
    os.makedirs(intention_dir, exist_ok=True)
    os.makedirs(generation_raw_dir, exist_ok=True)
    os.makedirs(intention_raw_dir, exist_ok=True)

    path_to_seed = f"/kaggle/input/scholawrite-seeds/{seed_name}.txt"


def main_openai():
    for seed_name in ["seed1"]:
        print(f"Working on {seed_name} with OpenAI API")
        setup_openai(seed_name)
        aggregate_iterative_writing_openai()
        print("-" * 100)

# %%
main_openai()
shutil.make_archive("/kaggle/working/all_outputs", 'zip', "/kaggle/working")

# %%
import shutil
shutil.make_archive("/kaggle/working/all_outputs", 'zip', "/kaggle/working")

# %%
def get_similar_llama(text1, text2, model, tokenizer):
    pipl = pipeline('feature-extraction', model=model, tokenizer=tokenizer)
    data = pipl(text1)
    data1 = torch.tensor(data)

    data = pipl(text2)
    data2 = torch.tensor(data)

    sentence_embedding1 = data1.mean(dim=1)
    sentence_embedding2 = data2.mean(dim=1)

    result = cosine_similarity(sentence_embedding1, sentence_embedding2, dim=1)

    return result


def main():
    seed_path = "../seeds"
    output_abs_path = "../outputs"
    outputs = ["llama8_meta_output", "llama8_SW_output", "gpt4o_output"]
    all_seeds = ["seed1", "seed2", "seed3", "seed4"]
    all_output = {}

    for output in outputs:
        all_output[output] = {}
        for seed in tqdm(all_seeds):
            try:
                path_to_seed = os.path.join(seed_path, f"{seed}.txt")
                path_to_folder = os.path.join(output_abs_path, output, seed, "generation/iter_generation_99.txt")

                with open(path_to_seed) as file:
                    seed_text = file.read()

                with open(path_to_folder) as file:
                    final_text = file.read()

                all_output[output][seed] = get_similar_llama(seed_text, final_text, model, tokenizer)
            except:
                continue

    print(all_output)

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

with open('/kaggle/input/scholawrite-seeds/seed1.txt', 'r', encoding='utf-8') as f_seed:
    seed_text = f_seed.read()

with open('/kaggle/working/seed1/generation/iter_generation_99.txt', 'r', encoding='utf-8') as f_final:
    final_text = f_final.read()

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([seed_text, final_text])

cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

print(f"Cosine similarity between seed and final output: {cos_sim:.4f}")

# %% [markdown]
# ## Experiment Records:
# - llama-3-8b-SW: Cosine similarity between seed and final output: 0.4023
# - llama-meta: Cosine similarity between seed and final output: 0.2571

# %%



