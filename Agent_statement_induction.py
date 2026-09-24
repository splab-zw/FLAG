from openai import OpenAI
import pickle
import os

# Initialize conversation history with system prompt
conversation_history = [
    {"role": "system", "content": "You are a helpful assistant."}
]

# Function to call LLM
def call_llm(prompt):
    global conversation_history
    try:
        client = OpenAI(api_key="YOUR_API_KEY", base_url="http://127.0.0.1:8000/v1")
        conversation_history.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model="deepseek",
            messages=conversation_history,
            stream=False
        )

        ai_response = response.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": ai_response})
        return ai_response
    except Exception as e:
        return f"Error: {str(e)}"

# Parse categorized factual statements from LLM output
def parse_categorized_statements(text):
    support, contradict, neutral = [], [], []
    section = None
    for line in text.strip().splitlines():
        l = line.strip()
        if not l:
            continue
        if l.lower().startswith("### support"):
            section = "support"
        elif l.lower().startswith("### contradict"):
            section = "contradict"
        elif l.lower().startswith("### neutral"):
            section = "neutral"
        elif l.startswith("-"):
            content = l.lstrip("- ").strip()
            if section == "support":
                support.append(content)
            elif section == "contradict":
                contradict.append(content)
            elif section == "neutral":
                neutral.append(content)
    return support, contradict, neutral

# Phase 1: Induce anchor factual statements from correct/incorrect answers
def induce_anchors_from_sets(correct_answers, incorrect_answers):
    correct_list = '\n'.join([f"{i+1}. {ans}" for i, ans in enumerate(correct_answers)])
    incorrect_list = '\n'.join([f"{i+1}. {ans}" for i, ans in enumerate(incorrect_answers)])

    prompt = f"""
You are a legal education expert specializing in Economic Law for intermediate-level professional exams.

You are given two sets of student answers.

Correct Answers:
{correct_list}

Incorrect Answers:
{incorrect_list}

Task: Extract atomic factual statements reflecting key Knowledge Points:
- Statements in correct answers -> SUPPORT
- Statements in incorrect answers suggesting misconceptions -> CONTRADICT
- Vague or unclear statements -> NEUTRAL

Requirements:
- Atomic, concise, unambiguous
- Follow this format:

### SUPPORT
- statement A

### CONTRADICT
- statement B

### NEUTRAL
- statement C
"""
    output = call_llm(prompt)
    print("🔄 Raw Output:\n", output)
    return parse_categorized_statements(output)

# Phase 2+3: Expand statements iteratively and cluster until convergence
def expand_until_converged(unlabeled_answers, support, contradict, neutral, batch_size: int = 10, max_rounds: int = 8):
    for t in range(max_rounds):
        print(f"\n🔁 Iteration {t + 1}")
        batch = unlabeled_answers[t * batch_size:(t + 1) * batch_size]
        if not batch:
            print("⚠️ No more samples.")
            break

        support_list = '\n'.join([f"- {s}" for s in support])
        contradict_list = '\n'.join([f"- {s}" for s in contradict])
        neutral_list = '\n'.join([f"- {s}" for s in neutral])
        answer_list = '\n'.join([f"{i + 1}. {a}" for i, a in enumerate(batch)])

        prompt = f"""
You are expanding an existing categorized set of factual statements based on new student answers.

Existing statements:
SUPPORT: core knowledge points
CONTRADICT: misconceptions
NEUTRAL: ambiguous/unclear

Task:
1. Analyze each student answer.
2. Extract new expressions of existing knowledge points.
3. Categorize as SUPPORT/CONTRADICT/NEUTRAL.
4. At most 3 statements per answer.

Existing SUPPORT:
{support_list}

Existing CONTRADICT:
{contradict_list}

Existing NEUTRAL:
{neutral_list}

New Student Answers:
{answer_list}

Output format:
Answer 1:
- [SUPPORT] ...
- [CONTRADICT] ...
- [NEUTRAL] ...

Answer 2:
...
"""
        output = call_llm(prompt)
        print("🔄 Expansion Output:\n", output)

        new_support, new_contradict, new_neutral = [], [], []

        for line in output.strip().splitlines():
            if "[SUPPORT]" in line:
                s = line.split("]", 1)[-1].strip()
                if s and s not in support and s not in new_support:
                    new_support.append(s)
            elif "[CONTRADICT]" in line:
                s = line.split("]", 1)[-1].strip()
                if s and s not in contradict and s not in new_contradict:
                    new_contradict.append(s)
            elif "[NEUTRAL]" in line:
                s = line.split("]", 1)[-1].strip()
                if s and s not in neutral and s not in new_neutral:
                    new_neutral.append(s)

        print(f"➕ New SUPPORT: {len(new_support)}, CONTRADICT: {len(new_contradict)}, NEUTRAL: {len(new_neutral)}")

        support += new_support
        contradict += new_contradict
        neutral += new_neutral

        # Review for completeness and clustering
        review_prompt = f"""
Review current categorized factual statements for completeness and redundancy.

SUPPORT:
{'\n'.join([f'- {s}' for s in support])}

CONTRADICT:
{'\n'.join([f'- {s}' for s in contradict])}

NEUTRAL:
{'\n'.join([f'- {s}' for s in neutral])}

Task:
- If statements sufficiently cover key knowledge points, reply: 'Sufficiently covered' and cluster statements by similarity, removing duplicates.
- Otherwise, reply: 'Not yet'
"""
        review_output = call_llm(review_prompt)
        print("🔍 Review Output:\n", review_output)

        if "sufficiently covered" in review_output.lower():
            print("✅ LLM says: Sufficiently covered. Stopping and parsing clusters.")

            def parse_clusters(text):
                from collections import defaultdict
                category = None
                clusters = defaultdict(list)
                cluster_id = 0

                for line in text.strip().splitlines():
                    l = line.strip()
                    if l.lower().startswith("### support"):
                        category = "support"
                        cluster_id = 0
                    elif l.lower().startswith("### contradict"):
                        category = "contradict"
                        cluster_id = 0
                    elif l.lower().startswith("### neutral"):
                        category = "neutral"
                        cluster_id = 0
                    elif l.lower().startswith("- cluster"):
                        cluster_id += 1
                    elif l.startswith("-") or l.startswith("\u2022"):
                        statement = l.lstrip("-\u2022 ").strip()
                        if category:
                            clusters[(category, cluster_id)].append(statement)

                grouped = {"support": [], "contradict": [], "neutral": []}
                for (cat, cid), members in clusters.items():
                    grouped[cat].append(members)
                return grouped

            return parse_clusters(review_output)
        else:
            print("🔄 LLM says: More coverage needed. Continue expanding...")

    # Return flat clusters if max rounds reached
    return {"support": [[s] for s in support], "contradict": [[s] for s in contradict],
            "neutral": [[s] for s in neutral]}

# Main controller function
def run_statement_induction(correct_answers, incorrect_answers, unlabeled_data):
    print("🔹 Phase 1: Inducing Anchors")
    support, contradict, neutral = induce_anchors_from_sets(correct_answers, incorrect_answers)
    print(f"Initial SUPPORT: {len(support)}, CONTRADICT: {len(contradict)}, NEUTRAL: {len(neutral)}")

    print("\n🔁 Phase 2+3: Expansion + Review")
    final_statements = expand_until_converged(unlabeled_data, support, contradict, neutral)

    # Print clustered statements
    print("\n✅ Final Clustered Statement Sets:")
    for category in ["support", "contradict", "neutral"]:
        print(f"\n### {category.upper()} ({len(final_statements[category])} clusters)")
        for idx, cluster in enumerate(final_statements[category], 1):
            print(f"  - Cluster {idx}:")
            for stmt in cluster:
                print(f"    • {stmt}")

    return final_statements

# Save flattened statements and cluster indices
def save_flattened_clusters(final_statements):
    flat_statements = {}
    cluster_indices = {}

    for category in ["support", "contradict", "neutral"]:
        flat_list = []
        cluster_ids = []
        for cluster_idx, cluster in enumerate(final_statements[category]):
            for statement in cluster:
                flat_list.append(statement)
                cluster_ids.append(cluster_idx)
        flat_statements[category] = flat_list
        cluster_indices[category] = cluster_ids

    with open(f"statements.pkl", "wb") as f:
        pickle.dump(flat_statements, f)

    with open(f"cluster_ids.pkl", "wb") as f:
        pickle.dump(cluster_indices, f)

    print(f"✅ Saved flattened statements and cluster indices to '*.pkl'")

# Entry point
if __name__ == "__main__":
    # Load placeholders for data files
    with open(f'PATH_TO_INSTANCES_PICKLE', 'rb') as f:
        instances = pickle.load(f)

    with open(f'PATH_TO_LABELS_PICKLE', 'rb') as f:
        labels = pickle.load(f)

    with open(f'PATH_TO_UNLABELED_PICKLE', 'rb') as f:
        unlabeled = pickle.load(f)

    correct_answers = [instances[i] for i in range(len(instances)) if labels[i] == 1]
    incorrect_answers = [instances[i] for i in range(len(instances)) if labels[i] == 0]

    final_statements = run_statement_induction(correct_answers, incorrect_answers, unlabeled)
    print(final_statements)

    save_flattened_clusters(final_statements)
