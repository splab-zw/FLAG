import numpy as np
from typing import List
import pickle

# -------------------------------
# Helper functions for label propagation and matching
# -------------------------------

def get_cluster_level_vectors(flat_matrix: List[List[int]], cluster_ids: List[int]) -> List[List[int]]:
    """
    Convert statement-level satisfaction vectors (N x M) to cluster-level vectors (N x C)
    - Each cluster is 1 if any statement in it is 1
    """
    N = len(flat_matrix)
    M = len(flat_matrix[0])
    assert len(cluster_ids) == M, f"cluster_ids must match statement dimension M={M}"
    C = max(cluster_ids) + 1  # Total clusters

    cluster_matrix = []
    for row in flat_matrix:
        cluster_row = [0] * C
        for j in range(M):
            if row[j] == 1:
                cluster_idx = cluster_ids[j]
                cluster_row[cluster_idx] = 1
        cluster_matrix.append(cluster_row)
    return cluster_matrix


def strict_match_by_cluster(cluster_matrix: List[List[int]], labels: List[int]) -> List[int]:
    """
    Strict matching on cluster-level vectors:
    - Assign label if any labeled template exactly matches
    """
    labeled_vecs = cluster_matrix[:len(labels)]
    unlabeled_vecs = cluster_matrix[len(labels):]
    matched_labels = [-1] * len(unlabeled_vecs)

    templates = {1: [], 0: []}
    for i, label in enumerate(labels):
        templates[label].append(labeled_vecs[i])

    for idx, vec in enumerate(unlabeled_vecs):
        matched = False
        for label in [1, 0]:
            for template in templates[label]:
                if vec == template:
                    matched_labels[idx] = label
                    matched = True
                    break
            if matched:
                break
    return matched_labels


def loose_match_by_cluster(
    cluster_matrix: List[List[int]],
    labels: List[int],
    cluster_structure: List[int]  # [num_support_clusters, num_contradict_clusters, num_neutral_clusters]
) -> List[int]:
    """
    Loose matching logic on cluster-level vectors:
    - Positive match: support ⊇ template.support and contradict ⊆ template.contradict
    - Negative match: support ⊆ template.support and contradict ⊇ template.contradict
    """
    K = len(labels)
    support_len, contradict_len, _ = cluster_structure

    all_support_sets = []
    all_contradict_sets = []

    for row in cluster_matrix:
        s_set = {i for i in range(support_len) if row[i] == 1}
        c_set = {i for i in range(support_len, support_len + contradict_len) if row[i] == 1}
        all_support_sets.append(s_set)
        all_contradict_sets.append(c_set)

    test_support_sets = all_support_sets[K:]
    test_contradict_sets = all_contradict_sets[K:]
    matched_labels = [-1] * len(test_support_sets)

    # Build templates
    positive_templates = [(all_support_sets[i], all_contradict_sets[i]) for i in range(K) if labels[i] == 1]
    negative_templates = [(all_support_sets[i], all_contradict_sets[i]) for i in range(K) if labels[i] == 0]

    for idx, (s_set, c_set) in enumerate(zip(test_support_sets, test_contradict_sets)):
        # Positive match
        for p_supp, p_contr in positive_templates:
            if p_supp.issubset(s_set) and c_set.issubset(p_contr):
                matched_labels[idx] = 1
                break
        # Negative match
        if matched_labels[idx] == -1:
            for n_supp, n_contr in negative_templates:
                if s_set.issubset(n_supp) and n_contr.issubset(c_set):
                    matched_labels[idx] = 0
                    break

    return matched_labels


def strict_match(satisfy_matrix, labels):
    """
    Strict matching at statement level: assign label if vector exactly matches any template
    """
    labeled_vecs = satisfy_matrix[:len(labels)]
    unlabeled_vecs = satisfy_matrix[len(labels):]
    matched_labels = [-1] * len(unlabeled_vecs)

    templates = {1: [], 0: []}
    for i, label in enumerate(labels):
        templates[label].append(labeled_vecs[i])

    for idx, vec in enumerate(unlabeled_vecs):
        matched = False
        for label in [1, 0]:
            for template in templates[label]:
                if vec == template:
                    matched_labels[idx] = label
                    matched = True
                    break
            if matched:
                break
    return matched_labels


def loose_match_from_matrix(
    satisfy_matrix: List[List[int]],
    labels: List[int],
    structure: List[int]
) -> List[int]:
    """
    Loose matching at statement level (support/contradict sets)
    """
    K = len(labels)
    support_len, contradict_len, _ = structure

    all_support_sets = []
    all_contradict_sets = []

    for row in satisfy_matrix:
        support_set = {i for i in range(support_len) if row[i] == 1}
        contradict_set = {i for i in range(support_len, support_len + contradict_len) if row[i] == 1}
        all_support_sets.append(support_set)
        all_contradict_sets.append(contradict_set)

    test_support_sets = all_support_sets[K:]
    test_contradict_sets = all_contradict_sets[K:]
    matched_labels = [-1] * len(test_support_sets)

    positive_templates = [(all_support_sets[i], all_contradict_sets[i]) for i in range(K) if labels[i] == 1]
    negative_templates = [(all_support_sets[i], all_contradict_sets[i]) for i in range(K) if labels[i] == 0]

    for idx, (s_set, c_set) in enumerate(zip(test_support_sets, test_contradict_sets)):
        for p_supp, p_contr in positive_templates:
            if p_supp.issubset(s_set) and c_set.issubset(p_contr):
                matched_labels[idx] = 1
                break
        if matched_labels[idx] == -1:
            for n_supp, n_contr in negative_templates:
                if s_set.issubset(n_supp) and n_contr.issubset(c_set):
                    matched_labels[idx] = 0
                    break
    return matched_labels


def propagate_labels_by_knn(current_labels: List[int], knn_index: List[List[int]], k: int = 3) -> List[int]:
    """
    Single-step kNN label propagation:
    - Only propagate if first k neighbors all have the same known label
    """
    new_labels = current_labels[:]
    N = len(new_labels)
    newly_labeled = 0

    print(f"\n🔁 Static kNN propagation with k={k} (strict mode)")

    for i in range(N):
        if new_labels[i] != -1:
            continue

        neighbors = knn_index[i][:k]
        neighbor_labels = [current_labels[j] for j in neighbors]

        if -1 in neighbor_labels:
            continue

        if all(l == 1 for l in neighbor_labels):
            new_labels[i] = 1
            newly_labeled += 1
        elif all(l == 0 for l in neighbor_labels):
            new_labels[i] = 0
            newly_labeled += 1

    print(f"✅ Labels propagated (strict consensus): {newly_labeled}")
    return new_labels


def calculate_metrics(pred_list, true_list):
    """
    Compute accuracy, recall, precision, F1 over predictions ignoring -1
    """
    filtered_pred = []
    filtered_true = []
    for p, t in zip(pred_list, true_list):
        if p != -1:
            filtered_pred.append(p)
            filtered_true.append(t)
    n = len(filtered_pred)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0, len(true_list)
    TP = sum((p == 1 and t == 1) for p, t in zip(filtered_pred, filtered_true))
    FP = sum((p == 1 and t == 0) for p, t in zip(filtered_pred, filtered_true))
    TN = sum((p == 0 and t == 0) for p, t in zip(filtered_pred, filtered_true))
    FN = sum((p == 0 and t == 1) for p, t in zip(filtered_pred, filtered_true))
    accuracy = (TP + TN) / n
    recall = TP / (TP + FN) if (TP + FN) else 0
    precision = TP / (TP + FP) if (TP + FP) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return accuracy, recall, precision, f1, n, len(true_list)


def save_template_set(
    texts: List[str],
    labels: List[int],
    statement_vectors: List[List[int]],
    cluster_ids: List[int],
    save_path: str
):
    """
    Save template set (text + label + statement + cluster vectors) to file
    """
    cluster_vectors = get_cluster_level_vectors(statement_vectors, cluster_ids)
    with open(save_path, "w") as f:
        for i, (text, label, svec, cvec) in enumerate(zip(texts, labels, statement_vectors, cluster_vectors)):
            f.write(f"Sample {i+1} | Label: {label}\n")
            f.write(f"Text: {text}\n")
            f.write(f"Statement Vector: {svec}\n")
            f.write(f"Cluster Vector: {cvec}\n\n")
    print(f"✅ Template set saved to {save_path}")


def save_error_analysis_with_cluster(
    texts: List[str],
    pred_labels: List[int],
    true_labels: List[int],
    statement_vectors: List[List[int]],
    cluster_ids: List[int],
    save_prefix: str
):
    """
    Save error analysis categorized by TP, TN, FP, FN, UNK
    """
    cluster_vectors = get_cluster_level_vectors(statement_vectors, cluster_ids)

    assert len(texts) == len(pred_labels) == len(true_labels) == len(statement_vectors)

    categories = {"TP": [], "TN": [], "FP": [], "FN": [], "UNK": []}
    for i, (text, pred, true, svec, cvec) in enumerate(zip(texts, pred_labels, true_labels, statement_vectors, cluster_vectors)):
        if pred == -1:
            categories["UNK"].append((text, svec, cvec, pred, true))
        elif pred == 1 and true == 1:
            categories["TP"].append((text, svec, cvec, pred, true))
        elif pred == 0 and true == 0:
            categories["TN"].append((text, svec, cvec, pred, true))
        elif pred == 1 and true == 0:
            categories["FP"].append((text, svec, cvec, pred, true))
        elif pred == 0 and true == 1:
            categories["FN"].append((text, svec, cvec, pred, true))

    for tag, samples in categories.items():
        with open(f"{save_prefix}_{tag}.txt", "w") as f:
            for idx, (text, svec, cvec, pred, true) in enumerate(samples):
                f.write(f"Sample {idx+1} | Pred: {pred} | True: {true} | Type: {tag}\n")
                f.write(f"Text: {text}\n")
                f.write(f"Statement Vector: {svec}\n")
                f.write(f"Cluster Vector: {cvec}\n\n")

    print(f"✅ Error breakdown written to {save_prefix}_*.txt")


# -------------------------------
# Main iterative label expansion
# -------------------------------

def iterative_label_expansion(
        satisfy_matrix,
        labels,
        tests_labels,
        structure,
        cluster_structure,
        knn_index,
        match_type="loose",
        k=5,
        max_rounds=10,
        use_cluster=True,
        cluster_ids=None
):
    """
    Iterative label expansion using template matching + kNN propagation
    """
    from copy import deepcopy

    assert match_type in ["strict", "loose"], "Invalid match_type"
    N = len(satisfy_matrix)
    K = len(labels)

    current_labels = labels + [-1] * (N - K)
    used_mask = [True] * K + [False] * (N - K)

    print(f"🚀 Starting iterative expansion using match_type = {match_type.upper()}, use_cluster = {use_cluster}")

    for round_id in range(max_rounds):
        print(f"\n🔁 Iteration {round_id + 1}")
        template_indices = [i for i, used in enumerate(used_mask) if used]
        template_labels = [current_labels[i] for i in template_indices]
        template_matrix = [satisfy_matrix[i] for i in template_indices]

        remaining_indices = [i for i in range(N) if not used_mask[i]]
        if not remaining_indices:
            print("✅ All samples used. Stopping.")
            break

        remaining_matrix = [satisfy_matrix[i] for i in remaining_indices]
        temp_matrix = template_matrix + remaining_matrix
        temp_labels = template_labels

        if use_cluster:
            assert cluster_ids is not None, "cluster_ids required if use_cluster=True"
            temp_cluster_matrix = get_cluster_level_vectors(temp_matrix, cluster_ids)
            if match_type == "strict":
                new_labels = strict_match_by_cluster(temp_cluster_matrix, temp_labels)
            else:
                new_labels = loose_match_by_cluster(temp_cluster_matrix, temp_labels, cluster_structure)
        else:
            if match_type == "strict":
                new_labels = strict_match(temp_matrix, temp_labels)
            else:
                new_labels = loose_match_from_matrix(temp_matrix, temp_labels, structure)

        match_count = 0
        for i, idx in enumerate(remaining_indices):
            if new_labels[i] != -1:
                current_labels[idx] = new_labels[i]
                used_mask[idx] = True
                match_count += 1

        print(f"🔍 Matching newly labeled samples: {match_count}")

        current_labels = propagate_labels_by_knn(current_labels, knn_index, k=k)

        propagate_count = 0
        for i in range(N):
            if not used_mask[i] and current_labels[i] != -1:
                used_mask[i] = True
                propagate_count += 1

        print(f"🌱 Propagation newly labeled samples: {propagate_count}")

        if match_count + propagate_count == 0:
            print("🚨 No new labels assigned this round. Converged.")
            break

    final_test_labels = current_labels[K:]
    acc, recall, prec, f1, used, total = calculate_metrics(final_test_labels, tests_labels)

    print(f"\n📊 Final Evaluation (after iteration):")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"#Correct/Sum: {acc*used}")
    print(f"Evaluated on {used} / {total} test samples")

    return current_labels, final_test_labels


# -------------------------------
# Execution main block
# -------------------------------
if __name__ == "__main__":
    type_ = 'Q2'

    # Load all input files (placeholders)
    with open(f'PATH_TO/{type_}-instances-labels.pkl','rb') as f:
        labels = pickle.load(f)
    with open(f'PATH_TO/{type_}-tests.pkl', 'rb') as f:
        tests = pickle.load(f)
    with open(f'PATH_TO/{type_}-test_labels.pkl','rb') as f:
        tests_labels = pickle.load(f)
    with open(f'PATH_TO/{type_}-instances-values.pkl','rb') as f:
        instances_values = pickle.load(f)
    with open(f'PATH_TO/{type_}-tests-values.pkl','rb') as f:
        tests_values = pickle.load(f)
    with open(f'PATH_TO/{type_}-statements.pkl','rb') as f:
        sta = pickle.load(f)
    with open(f'PATH_TO/{type_}-cluster_ids.pkl','rb') as f:
        cluster_ids = pickle.load(f)
    with open(f'PATH_TO/{type_}-knn_index.pkl','rb') as f:
        knn_index = pickle.load(f)

    # Build statements structure
    statements = []
    structure=[]
    for category in ["support", "contradict", "neutral"]:
        statements.extend(sta[category])
        structure.append(len(sta[category]))
    print(structure)

    # Combine instance and test values
    all_values = instances_values + tests_values

    # Flatten cluster ids across categories
    cluster_ids_flat = []
    offset = 0
    for category in ["support", "contradict", "neutral"]:
        local_ids = cluster_ids[category]
        max_local_id = max(local_ids) if local_ids else -1
        cluster_ids_flat.extend([cid + offset for cid in local_ids])
        offset += max_local_id + 1

    support_cluster_count = max(cluster_ids["support"]) + 1 if cluster_ids["support"] else 0
    contradict_cluster_count = max(cluster_ids["contradict"]) + 1 if cluster_ids["contradict"] else 0
    neutral_cluster_count = max(cluster_ids["neutral"]) + 1 if cluster_ids["neutral"] else 0
    cluster_structure = [support_cluster_count, contradict_cluster_count, neutral_cluster_count]

    # Iterative label expansion
    final_labels, predicted_test = iterative_label_expansion(
        satisfy_matrix=all_values,
        labels=labels,
        tests_labels=tests_labels,
        structure=structure,
        cluster_structure=cluster_structure,
        knn_index=knn_index,
        match_type="loose",
        k=8,
        max_rounds=10,
        use_cluster=True,
        cluster_ids=cluster_ids_flat
    )

    save_error_analysis_with_cluster(
        texts=tests,
        pred_labels=predicted_test,
        true_labels=tests_labels,
        statement_vectors=tests_values,
        cluster_ids=cluster_ids_flat,
        save_prefix=f"{type}-error"
    )

