# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# %% [markdown]
# # Data Processing

# %%
data1 = pd.read_parquet('/kaggle/input/scholawrite/all_sorted-00000-of-00002.parquet', engine='pyarrow')
data2 = pd.read_parquet('/kaggle/input/scholawrite/all_sorted-00001-of-00002.parquet', engine='pyarrow')

# %%
df = pd.concat([data1, data2], axis=0, ignore_index=True)
df.head()

# %%
project_tables = {project: group for project, group in df.groupby('project')}
for project, table in project_tables.items():
    project_tables[project] = table.sort_values(by='timestamp', ascending=True)
project_tables[1].head()

# %%
exp_data = pd.read_csv("/kaggle/input/experiment-data/model_outputs (1).csv")
exp_data.head()

# %%
model_tables = {}

for model_name, group_df in exp_data.groupby('model'):
    sorted_df = group_df.sort_values(by='index_in_file').reset_index(drop=True)
    model_tables[model_name] = sorted_df

for model_name, df in model_tables.items():
    print(f"Model: {model_name}")

# %% [markdown]
# # Human Behaviorial Pattern Mining
# ## LCS Analysis
# ### Elementary Continuous Intention Analysis

# %%
def longest_contiguous_label_run_np(label_series):
    arr = np.array(label_series)
    change_points = np.where(arr[1:] != arr[:-1])[0] + 1
    change_points = np.concatenate(([0], change_points, [len(arr)]))
    run_lengths = np.diff(change_points)
    run_labels = arr[change_points[:-1]]
    max_idx = np.argmax(run_lengths)
    return run_labels[max_idx], run_lengths[max_idx]

# %%
for i in range(1, 6):
    table = project_tables[i]
    
    label_series = table['label']
    
    longest_label, length = longest_contiguous_label_run_np(label_series)
    print(f"longest label: {longest_label}; longest sequence length: {length}")

# %%
for model_name, df in model_tables.items():
    labels = df['label']
    longest_label, length = longest_contiguous_label_run_np(labels)
    print(f"{model_name}: longest label: {longest_label}; longest sequence length: {length}")

# %% [markdown]
# ### Sliding Window

# %%
from collections import Counter

def most_common_subsequence(label_series, window_size=50):
    sequence_list = list(label_series)
    n = len(sequence_list)
    
    if n < window_size:
        return None, 0

    subsequences = [tuple(sequence_list[i:i+window_size]) for i in range(n - window_size + 1)]

    counter = Counter(subsequences)
    most_common_seq, freq = counter.most_common(1)[0]
    
    return most_common_seq, freq

# %%
for i in range(1, 6):
    table = project_tables[i]
    
    label_series = table['label']
    
    longest_label, length = most_common_subsequence(label_series)
    print(f"longest common subsequence: {longest_label}; longest frequency length: {length}")

# %% [markdown]
# ### Sliding Window(diff-val constraint version)

# %%
def most_frequent_contiguous_subsequence_with_constraint(label_sequence, window_size=3, min_support=3, diff_constraint=1):
    n = len(label_sequence)
    if n < window_size:
        return None, 0
    windows = [
        tuple(label_sequence[i:i+window_size])
        for i in range(n - window_size + 1)
        if len(set(label_sequence[i:i+window_size])) > diff_constraint
    ]

    counter = Counter(windows)
    counter = {k: v for k, v in counter.items() if v >= min_support}

    if not counter:
        return None, 0

    most_common = max(counter.items(), key=lambda x: (x[1], len(x[0])))
    return most_common

# %% [markdown]
# #### Human Behaviorial Data
# 
# Project 1
# - differ_val=0: longest common subsequence: (('Text Production', 'Text Production', 'Text Production'), 8655)
# - differ_val=1: longest common subsequence: (('Coherence', 'Text Production', 'Text Production'), 67)
# - differ_val=2: longest common subsequence: (('Clarity', 'Coherence', 'Text Production'), 7)
# - differ_val=3: longest common subsequence: (None, 0)

# %%
for i in range(1, 6):
    table = project_tables[i]
    
    label_series = table['label']
    
    longest_label = most_frequent_contiguous_subsequence_with_constraint(label_series, 3, 3, 0)
    print(f"longest common subsequence: {longest_label}")

# %%
for i in range(1, 6):
    table = project_tables[i]
    
    label_series = table['label']
    
    longest_label = most_frequent_contiguous_subsequence_with_constraint(label_series)
    print(f"longest common subsequence: {longest_label}")

# %%
for i in range(1, 6):
    table = project_tables[i]
    
    label_series = table['label']
    
    longest_label = most_frequent_contiguous_subsequence_with_constraint(label_series, 3, 3, 2)
    print(f"longest common subsequence: {longest_label}")

# %%
for i in range(1, 6):
    table = project_tables[i]
    
    label_series = table['label']
    
    longest_label = most_frequent_contiguous_subsequence_with_constraint(label_series, 3, 3, 3)
    print(f"longest common subsequence: {longest_label}")

# %% [markdown]
# #### LLM Behaviorial Data

# %%
!pip install ace_tools

# %%
results = []

for model_name, df in model_tables.items():
    labels = df['label'].tolist()
    for diff_val in range(3):
        longest_label = most_frequent_contiguous_subsequence_with_constraint(labels, 3, 3, diff_val)
        results.append({
            "model": model_name,
            "diff_val": diff_val,
            "longest_subsequence": longest_label
        })

results_df = pd.DataFrame(results)
results_df

# %%
def clean_subsequence(value):
    if isinstance(value, tuple) and isinstance(value[0], (list, tuple)):
        return ", ".join(value[0])
    return ""

def extract_frequency(value):
    if isinstance(value, tuple) and isinstance(value[1], int):
        return value[1]
    return 0

results_df["cleaned_subsequence"] = results_df["longest_subsequence"].apply(clean_subsequence)
results_df["frequency"] = results_df["longest_subsequence"].apply(extract_frequency)

# %%
results_df.drop(columns=["longest_subsequence"], inplace=True)
results_df.rename(columns={"cleaned_subsequence": "longest_subsequence"}, inplace=True)

# %%
results_df

# %%
output_path = "/kaggle/working/final_subsequence_results.csv"
results_df.to_csv(output_path, index=False)

# %% [markdown]
# ## Hidden Markov Models

# %%
!pip install hmmlearn

# %%
import hmmlearn
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder

# %%
project_tables[1]['label']

# %%
def hmm_modeling(labels):
    le = LabelEncoder()
    encoded_obs = le.fit_transform(labels).reshape(-1, 1)
    
    n_unique_obs = len(le.classes_)
    max_states = 10
    
    bics = []
    models = []
    
    for n in range(2, max_states + 1):
        model = hmm.MultinomialHMM(n_components=n, n_iter=100, random_state=42)
        model.fit(encoded_obs)
    
        logL = model.score(encoded_obs)
    
        p = (n - 1) + n * (n - 1) + n * (n_unique_obs - 1)
        bic = -2 * logL + p * np.log(len(encoded_obs))
    
        bics.append(bic)
        models.append(model)
    
    optimal_n = np.argmin(bics) + 2
    best_model = models[optimal_n - 2]
    
    print(f"Optimal number of hidden states (BIC): {optimal_n}")

    logprob, hidden_states = best_model.decode(encoded_obs)
    
    plt.plot(range(2, max_states + 1), bics, marker='o')
    plt.xlabel("Number of hidden states")
    plt.ylabel("BIC")
    plt.title("Model selection using BIC")
    plt.grid(True)
    plt.show()
    
    return {
        'best_model': best_model,
        'optimal_n': optimal_n,
        'label_encoder': le,
        'encoded_obs': encoded_obs,
        'hidden_states': hidden_states,
        'bics': bics
    }

# %%
results = hmm_modeling(project_tables[1]['label'])

# %%
results = []
for i in range(1, 6):
    result = hmm_modeling(project_tables[i]['label'])
    results.append(result)

# %%
for i, result in enumerate(results, start=1):
    model = result['best_model']
    transmat = model.transmat_
    print(f"\n===== Model {i}: Transition Matrix (n_components = {result['optimal_n']}) =====")
    print(np.round(transmat, 3))

# %%
from collections import Counter, defaultdict

def dominant_label_transition(result, label_series):
    model = result['best_model']
    hidden_states = result['hidden_states']
    transmat = model.transmat_
    n_states = result['optimal_n']
    
    dominant_labels = {}
    for s in range(n_states):
        indices = np.where(hidden_states == s)[0]
        most_common = Counter(label_series.iloc[indices]).most_common(1)[0][0]
        dominant_labels[s] = most_common

    label_transitions = defaultdict(float)
    for i in range(n_states):
        for j in range(n_states):
            li = dominant_labels[i]
            lj = dominant_labels[j]
            label_transitions[(li, lj)] += transmat[i, j]

    print("Dominant Label Transitions:")
    for (from_label, to_label), prob in label_transitions.items():
        if prob > 0.05:  
            print(f"{from_label} → {to_label}: {prob:.3f}")

# %% [markdown]
# # Markov Chain Analysis
# ## First-order Markov Chain Construction
# ### Human Behaviors

# %%
def build_label_transition_matrix(labels):
    transitions = defaultdict(lambda: defaultdict(int))
    
    for i in range(len(labels) - 1):
        from_label = labels.iloc[i]
        to_label = labels.iloc[i + 1]
        transitions[from_label][to_label] += 1

    transition_df = pd.DataFrame(transitions).fillna(0).T
    transition_df = transition_df.div(transition_df.sum(axis=1), axis=0)

    return transition_df

# %%
transmit_matrices = []
for i in range(1, 6):
    df = build_label_transition_matrix(project_tables[i]['label'])
    transmit_matrices.append(df)
    # print(df)

# %%
all_labels = set()
for matrix in transmit_matrices:
    all_labels.update(matrix.index)
    all_labels.update(matrix.columns)

sorted_labels = sorted(all_labels)

aligned_matrices = []
for matrix in transmit_matrices:
    aligned = matrix.reindex(index=sorted_labels, columns=sorted_labels).fillna(0)
    aligned_matrices.append(aligned)


for i, matrix in enumerate(aligned_matrices, start=1):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", square=True)
    plt.title(f"Label Transition Probability Matrix - Project {i}")
    plt.xlabel("To Label")
    plt.ylabel("From Label")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# %%
total_counts = sum(
    matrix.mul(matrix.sum(axis=1), axis=0)
    for matrix in aligned_matrices
)

generalized_matrix = total_counts.div(total_counts.sum(axis=1), axis=0)

# %%
plt.figure(figsize=(10, 8))
sns.heatmap(generalized_matrix, annot=True, fmt=".2f", cmap="Blues", square=True)
plt.title("Generalized Label Transition Probability Matrix (All Projects)")
plt.xlabel("To Label")
plt.ylabel("From Label")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Model Behaviors

# %%
trans_matrices = []
for model_name, df in model_tables.items():
    trans_mat = build_label_transition_matrix(df['label'])
    trans_matrices.append({"name":model_name, "transition_matrix": trans_mat})

# %%
for item in trans_matrices:
    mat = item["transition_matrix"]
    aligned = mat.reindex(index=sorted_labels, columns=sorted_labels).fillna(0)
    item["transition_matrix"] = aligned

# %%
for item in trans_matrices:
    name = item["name"]
    mat = item["transition_matrix"]

    plt.figure(figsize=(10, 8))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="Blues", square=True)
    plt.title(f"{name} Transition Probability Matrix")
    plt.xlabel("To Label")
    plt.ylabel("From Label")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# %%
def simulate_markov_chain(P_df, label_order, start_label=None, steps=20, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    labels = label_order
    P = P_df.loc[labels, labels].values

    if start_label is None:
        current_index = np.random.choice(len(labels))
    else:
        current_index = labels.index(start_label)

    sequence = [labels[current_index]]

    for _ in range(steps - 1):
        probs = P[current_index]
        next_index = np.random.choice(len(labels), p=probs)
        sequence.append(labels[next_index])
        current_index = next_index

    return sequence

# %%
project_tables[1].shape

# %% [markdown]
# ## First-order Markov Simulation

# %%
sequence = simulate_markov_chain(
    P_df=generalized_matrix,
    label_order=sorted_labels,
    start_label="Text Production",
    steps=100,
    random_state=42
)

df = pd.DataFrame({'simulated_label': sequence})

df.to_csv('/kaggle/working/simulated_sequence.csv', index=False)

print("saved to /kaggle/working/simulated_sequence.csv")

# %%
label_series = pd.Series(sequence)
    
longest_label = most_frequent_contiguous_subsequence_with_constraint(label_series, 3, 3, 0)
print(f"longest common subsequence: {longest_label}")
longest_label = most_frequent_contiguous_subsequence_with_constraint(label_series, 3, 3, 1)
print(f"longest common subsequence: {longest_label}")
longest_label = most_frequent_contiguous_subsequence_with_constraint(label_series, 3, 3, 2)
print(f"longest common subsequence: {longest_label}")

# %% [markdown]
# # Sequential Comparison and Alignment

# %% [markdown]
# ## Top-k Common Subsequence

# %%
def extract_fixed_length_subsequences(seq, n):
    return [tuple(seq[i:i+n]) for i in range(len(seq) - n + 1)]

def top_k_common_fixed_subsequences(seq1, seq2, n=3, k=5):
    subseq1 = extract_fixed_length_subsequences(seq1, n)
    subseq2 = extract_fixed_length_subsequences(seq2, n)

    count1 = Counter(subseq1)
    count2 = Counter(subseq2)

    common_keys = set(count1.keys()).intersection(count2.keys())

    common_subs = []
    for subseq in common_keys:
        freq1 = count1[subseq]
        freq2 = count2[subseq]
        total = freq1 + freq2
        common_subs.append((subseq, freq1, freq2, total))

    common_subs.sort(key=lambda x: x[3], reverse=True)

    return common_subs[:k]

# %%
top_seqs = top_k_common_subsequences(sequence, project_tables[1]['label'].tolist())
for s in top_seqs:
    print(list(s))

# %% [markdown]
# ## Alignment
# ### LCS Similarity

# %%
from difflib import SequenceMatcher

def lcs_similarity(seq1, seq2):
    matcher = SequenceMatcher(None, seq1, seq2)
    lcs_len = matcher.find_longest_match(0, len(seq1), 0, len(seq2)).size
    return lcs_len / min(len(seq1), len(seq2))

# %%
# data storage
def calculate_similarity_set(project_index, sim_func):
    sim_storage = []
    for model_name, df in model_tables.items():
            sim = sim_func(project_tables[project_index]['label'].tolist(), df['label'].tolist())
            sim_storage.append({"project":project_index, "model":model_name, "similarity":sim})
            print(f"Project {project_index} - an LCS Similarity Comparison between {model_name} and Human: {sim}")
    return sim_storage

# %%
lcs_sim_storage = calculate_similarity_set(1, lcs_similarity)

# %%
lcs_similarity(project_tables[1]['label'].tolist(), sequence)

# %% [markdown]
# ### Lavenshtein Distance Similarity

# %%
!pip install Levenshtein

# %%
import Levenshtein

def list_levenshtein_distance(seq1, seq2):
    s1 = ' '.join(seq1)
    s2 = ' '.join(seq2)
    dist = Levenshtein.distance(s1, s2)
    norm = dist / max(len(s1), len(s2))  
    return 1 - norm  

# %%
levenshtein_sim_storage = calculate_similarity_set(1, list_levenshtein_distance)

# %%
list_levenshtein_distance(project_tables[1]['label'].tolist(), sequence)

# %%



