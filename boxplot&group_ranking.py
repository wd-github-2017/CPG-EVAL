import os
import re
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# ============================================================================
# Accuracy Distributions Across Tasks and Subject Groups
# Model at the "item x subject" level using mixed-effects model (LMM & GLMM)
# Goal: Test whether the ranking order of groups is statistically significant
# ============================================================================

print("=" * 80)
print("Accuracy Distributions Across Tasks and Subject Groups")
print("=" * 80)

# Convergence warnings are common; we only log them here and do not terminate the process
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ========= Configuration =========
EXCEL_FILE = "CPG-EVAL-core&results.csv"

RUN_TASKS = [
    "SINGLE", 
    "BATCH", 
    "SIM-GRA", 
    "CAT-GRA", 
    "CON-INS"
    ]

OUTPUT_DIR = "group_ranking_mixed-effects_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Plotting Toggle
PLOT_BOXPLOT = True

# Groups
GROUPS = {
    "Humans": ["Hum-1", "Hum-2", "Hum-3", "Hum-4"],
    "larger-scale models": [
        "Doubao-1-5-pro-32k-250115",
        "GPT4o-240806",
        "DeepSeek-v3_250324",
        "Qwen2.5-Max-250409",
    ],
    "semi-larger-scale models": [
        "Doubao-1-5-lite-32k-250115",
        "Qwen2.5-72B-Instruct",
        "GPT-4o-mini-2024-07-18",
    ],
    "smaller-scale models": [
        "Qwen2.5-7B-Instruct",
        "glm-4-9b-chat",
        "Llama-3.1-8B-instruct",
        "internlm2_5-7b-chat",
    ],
}

GROUP_ORDER = {
    "Humans": 1,
    "larger-scale models": 2,
    "semi-larger-scale models": 3,
    "smaller-scale models": 4,
}


def get_sig_level(p):
    if p is None or pd.isna(p):
        return "N/A"
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


def task_from_qbank(qbank: str):
    if "SINGLE" in qbank:
        return "SINGLE"
    if "BATCH" in qbank:
        return "BATCH"
    if "SIM-GRA" in qbank:
        return "SIM-GRA"
    if "CAT-GRA" in qbank:
        return "CAT-GRA"
    if "CON-INS" in qbank:
        return "CON-INS"
    return None


def subtask_from_qbank(qbank: str, task: str):
    qb = str(qbank).upper()
    if task == "SINGLE":
        if "SINGLE-T" in qb:
            return "SINGLE-T"
        if "SINGLE-F" in qb:
            return "SINGLE-F"
        return "SINGLE"
    if task == "BATCH":
        if "BATCH-T" in qb:
            return "BATCH-T"
        if "BATCH-F" in qb:
            return "BATCH-F"
        return "BATCH"
    if task == "CON-INS":
        if re.search(r"F\*?10", qb):
            return "CON-INS F*10"
        if re.search(r"T\*?5.*F\*?5", qb) or "T5F5" in qb:
            return "CON-INS T5F5"
        return "CON-INS"
    return task


def extract_tf_list(text, max_count):
    if pd.isna(text):
        return []
    matches = re.findall(r"[TF]", str(text).upper())
    return matches[:max_count]


def get_correct_answer(row, task):
    qbank = str(row.get("QuestionBank", ""))
    if task == "SINGLE":
        if "SINGLE-T" in qbank:
            return "T"
        if "SINGLE-F" in qbank:
            return "F"
        return None
    if task == "BATCH":
        if "BATCH-T" in qbank:
            return "T"
        if "BATCH-F" in qbank:
            return "F"
        return None
    if task in ["SIM-GRA", "CAT-GRA"]:
        ans = row.get("answer", None)
        val = str(ans).strip() if pd.notna(ans) else None
        return val if val else None
    if task == "CON-INS":
        ans = row.get("answer", None)
        if pd.isna(ans):
            return None
        return extract_tf_list(ans, max_count=10)
    return None


def score_response(task, correct_answer, response):
    """Scoring function for single item types (0/1). BATCH/CON-INS are treated by accuracy and do not use this function."""
    if pd.isna(response):
        return np.nan

    resp = str(response).strip()

    if task == "SINGLE":
        return 1 if resp.upper() == str(correct_answer).strip().upper() else 0

    if task == "SIM-GRA":
        return 1 if str(correct_answer).strip() in resp else 0

    if task == "CAT-GRA":
        # Consistently 1, inconsistent 0. Using inclusion match to handle verbose model output.
        ans_str = str(correct_answer).strip()
        return 1 if ans_str in resp else 0

    return np.nan


def one_sided_p_greater(z):
    return 1 - norm.cdf(z)


def batch_accuracy(correct_answer, response, total_parts=9, use_answered_only=False):
    correct_char = str(correct_answer).strip().upper()
    resp_list = extract_tf_list(response, max_count=total_parts) if pd.notna(response) else []
    if use_answered_only and len(resp_list) == 0:
        return None
    denom = len(resp_list) if use_answered_only else total_parts
    correct_count = 0
    for i in range(denom):
        resp_val = resp_list[i] if i < len(resp_list) else None
        if resp_val is not None and resp_val.upper() == correct_char:
            correct_count += 1
    return correct_count / float(denom) if denom > 0 else None


def con_ins_accuracy(correct_answer_list, response, total_parts=10, use_answered_only=False):
    answer_list = correct_answer_list if isinstance(correct_answer_list, list) else []
    answer_list = answer_list[:total_parts] + [None] * max(0, total_parts - len(answer_list))
    resp_list = extract_tf_list(response, max_count=total_parts) if pd.notna(response) else []
    if use_answered_only and len(resp_list) == 0:
        return None
    denom = len(resp_list) if use_answered_only else total_parts
    correct_count = 0
    for i in range(denom):
        resp_val = resp_list[i] if i < len(resp_list) else None
        ans_val = answer_list[i]
        if resp_val is not None and ans_val is not None and resp_val.upper() == ans_val.upper():
            correct_count += 1
    return correct_count / float(denom) if denom > 0 else None


def build_long_df(df):
    records = []
    for idx, row in df.iterrows():
        qbank = str(row.get("QuestionBank", ""))
        task = task_from_qbank(qbank)
        if task is None or task not in RUN_TASKS:
            continue

        correct_answer = get_correct_answer(row, task)
        if correct_answer is None:
            continue

        question_id = row.get("UID", None)
        if pd.isna(question_id) or question_id == "":
            question_id = f"Q_{idx}"

        for group_name, subjects in GROUPS.items():
            for subject in subjects:
                if subject not in df.columns:
                    continue
                response = row.get(subject, None)
                if group_name == "Humans" and (pd.isna(response) or str(response).strip() == ""):
                    continue
                # BATCH/CON-INS scoring by cell accuracy (0~1)
                if task == "BATCH":
                    score = batch_accuracy(
                        correct_answer,
                        response,
                        total_parts=9,
                        use_answered_only=(group_name == "Humans"),
                    )
                    records.append(
                        {
                            "question_id": str(question_id),
                            "QuestionBank": qbank,
                            "task": task,
                            "subject": subject,
                            "group": group_name,
                            "group_order": GROUP_ORDER[group_name],
                            "score": score,
                        }
                    )
                elif task == "CON-INS":
                    answer_list = correct_answer if isinstance(correct_answer, list) else []
                    score = con_ins_accuracy(
                        answer_list,
                        response,
                        total_parts=10,
                        use_answered_only=(group_name == "Humans"),
                    )
                    records.append(
                        {
                            "question_id": str(question_id),
                            "QuestionBank": qbank,
                            "task": task,
                            "subject": subject,
                            "group": group_name,
                            "group_order": GROUP_ORDER[group_name],
                            "score": score,
                        }
                    )
                else:
                    score = score_response(task, correct_answer, response)
                    if pd.isna(score):
                        continue
                    records.append(
                        {
                            "question_id": str(question_id),
                            "QuestionBank": qbank,
                            "task": task,
                            "subject": subject,
                            "group": group_name,
                            "group_order": GROUP_ORDER[group_name],
                            "score": score,
                        }
                    )
    return pd.DataFrame(records)


def run_glmm_for_task(df_task, task):
    if len(df_task) == 0:
        print(f"  [Skip] {task} No valid data")
        return None, []

    print(f"  Observations: {len(df_task)}; Items: {df_task['question_id'].nunique()}; Subjects: {df_task['subject'].nunique()}")

    df_task = df_task.copy()
    df_task["correct"] = df_task["score"].astype(int)

    # Linear Order Model (GLMM)
    try:
        model = BinomialBayesMixedGLM.from_formula(
            "correct ~ group_order",
            {
                "item_re": "0 + C(question_id)",
                "subj_re": "0 + C(subject)",
            },
            data=df_task,
        )
        res = model.fit_vb()
        fe_mean = res.fe_mean
        fe_sd = res.fe_sd
        intercept_mean, slope_mean = float(fe_mean[0]), float(fe_mean[1])
        intercept_sd, slope_sd = float(fe_sd[0]), float(fe_sd[1])
        z_slope = slope_mean / slope_sd if slope_sd > 0 else np.nan
        p_slope = 2 * (1 - norm.cdf(abs(z_slope))) if not pd.isna(z_slope) else np.nan
        order_consistent = (slope_mean < 0) and (p_slope < 0.05)

        print(f"  GLMM Order Model: slope={slope_mean:.4f}, p={p_slope:.6e} {get_sig_level(p_slope)}")

        # Categorical Group Model + Adjacent Differences
        model_cat = BinomialBayesMixedGLM.from_formula(
            "correct ~ C(group, Treatment(reference='Humans'))",
            {
                "item_re": "0 + C(question_id)",
                "subj_re": "0 + C(subject)",
            },
            data=df_task,
        )
        res_cat = model_cat.fit_vb()
        exog_names = list(model_cat.exog_names)
        fe_mean_cat = res_cat.fe_mean
        fe_sd_cat = res_cat.fe_sd
        name_to_idx = {n: i for i, n in enumerate(exog_names)}

        def get_beta(cat):
            if cat == "Humans":
                return 0.0, 0.0
            key = f"C(group, Treatment(reference='Humans'))[T.{cat}]"
            if key not in name_to_idx:
                key = f"C(group)[T.{cat}]"
            return float(fe_mean_cat[name_to_idx[key]]), float(fe_sd_cat[name_to_idx[key]])

        beta_large, sd_large = get_beta("larger-scale models")
        beta_semi, sd_semi = get_beta("semi-larger-scale models")
        beta_small, sd_small = get_beta("smaller-scale models")

        adjacent = []
        def add_adj(label, diff, se):
            z = diff / se if se and se > 0 else np.nan
            p1 = one_sided_p_greater(z) if not pd.isna(z) else np.nan
            ok = (diff > 0) and (p1 < 0.05)
            adjacent.append({
                "Comparison": label,
                "Diff_Eta": diff,
                "SE_Approx": se,
                "Z_Approx": z,
                "P_OneSided": p1,
                "Significant_OneSided_0.05": ok,
            })
            print(f"    {label}: diff={diff:.4f}, p(one-sided)={p1:.6e} {get_sig_level(p1)}")

        add_adj("Humans > larger-scale", -beta_large, sd_large)
        add_adj(
            "larger-scale > semi-larger-scale",
            beta_large - beta_semi,
            float(np.sqrt(sd_large ** 2 + sd_semi ** 2)),
        )
        add_adj(
            "semi-larger-scale > smaller-scale",
            beta_semi - beta_small,
            float(np.sqrt(sd_semi ** 2 + sd_small ** 2)),
        )

        strict_ok = all(a["Significant_OneSided_0.05"] for a in adjacent)

        summary = {
            "Task": task,
            "Model": "GLMM-Binomial",
            "N_Obs": int(len(df_task)),
            "N_Items": int(df_task["question_id"].nunique()),
            "N_Subjects": int(df_task["subject"].nunique()),
            "Slope_Mean": slope_mean,
            "Slope_SD": slope_sd,
            "Slope_Z": z_slope,
            "Slope_P": p_slope,
            "Order_Consistent": bool(order_consistent),
            "Strict_Order_Consistent": bool(strict_ok),
        }

        return summary, adjacent

    except ValueError as e:
        print(f"  [Error] GLMM fitting failed for {task}: {e}")
        return None, []
    except Exception as e:
        print(f"  [Error] Unexpected error in GLMM for {task}: {e}")
        return None, []


def run_lmm_for_task(df_task, task):
    if len(df_task) == 0:
        print(f"  [Skip] {task} No valid data")
        return None, []

    print(f"  Observations: {len(df_task)}; Items: {df_task['question_id'].nunique()}; Subjects: {df_task['subject'].nunique()}")

    df_task = df_task.copy()
    df_task["score"] = df_task["score"].astype(float)

    # Linear Order Model (LMM)
    model = MixedLM.from_formula(
        "score ~ group_order",
        groups="question_id",
        vc_formula={"subj_re": "0 + C(subject)"},
        data=df_task,
    )
    res = model.fit(reml=False)
    fe_mean = res.fe_params
    fe_sd = res.bse_fe
    intercept_mean, slope_mean = float(fe_mean["Intercept"]), float(fe_mean["group_order"])
    intercept_sd, slope_sd = float(fe_sd["Intercept"]), float(fe_sd["group_order"])
    z_slope = slope_mean / slope_sd if slope_sd > 0 else np.nan
    p_slope = 2 * (1 - norm.cdf(abs(z_slope))) if not pd.isna(z_slope) else np.nan
    order_consistent = (slope_mean < 0) and (p_slope < 0.05)

    print(f"  LMM Order Model: slope={slope_mean:.4f}, p={p_slope:.6e} {get_sig_level(p_slope)}")

    # Categorical Group Model + Adjacent Differences (LMM)
    model_cat = MixedLM.from_formula(
        "score ~ C(group, Treatment(reference='Humans'))",
        groups="question_id",
        vc_formula={"subj_re": "0 + C(subject)"},
        data=df_task,
    )
    res_cat = model_cat.fit(reml=False)
    fe_mean_cat = res_cat.fe_params
    fe_sd_cat = res_cat.bse_fe
    name_to_idx = {n: i for i, n in enumerate(fe_mean_cat.index.tolist())}

    def get_beta(cat):
        if cat == "Humans":
            return 0.0, 0.0
        key = f"C(group, Treatment(reference='Humans'))[T.{cat}]"
        if key not in name_to_idx:
            key = f"C(group)[T.{cat}]"
        return float(fe_mean_cat.iloc[name_to_idx[key]]), float(fe_sd_cat.iloc[name_to_idx[key]])

    beta_large, sd_large = get_beta("larger-scale models")
    beta_semi, sd_semi = get_beta("semi-larger-scale models")
    beta_small, sd_small = get_beta("smaller-scale models")

    adjacent = []

    def add_adj(label, diff, se):
        z = diff / se if se and se > 0 else np.nan
        p1 = one_sided_p_greater(z) if not pd.isna(z) else np.nan
        ok = (diff > 0) and (p1 < 0.05)
        adjacent.append({
            "Comparison": label,
            "Diff_Eta": diff,
            "SE_Approx": se,
            "Z_Approx": z,
            "P_OneSided": p1,
            "Significant_OneSided_0.05": ok,
        })
        print(f"    {label}: diff={diff:.4f}, p(one-sided)={p1:.6e} {get_sig_level(p1)}")

    add_adj("Humans > larger-scale", -beta_large, sd_large)
    add_adj(
        "larger-scale > semi-larger-scale",
        beta_large - beta_semi,
        float(np.sqrt(sd_large ** 2 + sd_semi ** 2)),
    )
    add_adj(
        "semi-larger-scale > smaller-scale",
        beta_semi - beta_small,
        float(np.sqrt(sd_semi ** 2 + sd_small ** 2)),
    )

    strict_ok = all(a["Significant_OneSided_0.05"] for a in adjacent)

    summary = {
        "Task": task,
        "Model": "LMM",
        "N_Obs": int(len(df_task)),
        "N_Items": int(df_task["question_id"].nunique()),
        "N_Subjects": int(df_task["subject"].nunique()),
        "Slope_Mean": slope_mean,
        "Slope_SD": slope_sd,
        "Slope_Z": z_slope,
        "Slope_P": p_slope,
        "Order_Consistent": bool(order_consistent),
        "Strict_Order_Consistent": bool(strict_ok),
    }

    return summary, adjacent


def build_score_table(df):
    subjects = [s for group in GROUPS.values() for s in group]
    rows = []
    for idx, row in df.iterrows():
        qbank = str(row.get("QuestionBank", ""))
        task = task_from_qbank(qbank)
        if task is None or task not in RUN_TASKS:
            continue
        correct_answer = get_correct_answer(row, task)
        if correct_answer is None:
            continue

        question_id = row.get("UID", None)
        if pd.isna(question_id) or question_id == "":
            question_id = f"Q_{idx}"

        out_row = {
            "UID": str(question_id),
            "task": task,
        }
        for subject in subjects:
            response = row.get(subject, None)
            if subject in GROUPS.get("Humans", []) and (pd.isna(response) or str(response).strip() == ""):
                out_row[subject] = np.nan
                continue
            if task == "BATCH":
                score = batch_accuracy(
                    correct_answer,
                    response,
                    total_parts=9,
                    use_answered_only=(subject in GROUPS.get("Humans", [])),
                )
            elif task == "CON-INS":
                score = con_ins_accuracy(
                    correct_answer,
                    response,
                    total_parts=10,
                    use_answered_only=(subject in GROUPS.get("Humans", [])),
                )
            else:
                score = score_response(task, correct_answer, response)
            out_row[subject] = score
        rows.append(out_row)
    return pd.DataFrame(rows)


def build_subject_task_accuracy_table(df_long):
    if len(df_long) == 0:
        return pd.DataFrame()

    df_stats = df_long.copy()
    df_stats["subtask"] = df_stats.apply(
        lambda r: subtask_from_qbank(r.get("QuestionBank", ""), r.get("task", "")),
        axis=1,
    )

    task_acc = (
        df_stats.groupby(["subject", "group", "subtask"], as_index=False)["score"]
        .mean()
        .rename(columns={"score": "accuracy"})
    )

    pivot = (
        task_acc.pivot_table(
            index=["subject", "group"],
            columns="subtask",
            values="accuracy",
            aggfunc="mean",
        )
        .reset_index()
    )

    overall = (
        df_long.groupby(["subject", "group"], as_index=False)["score"]
        .mean()
        .rename(columns={"score": "overall_accuracy"})
    )

    merged = pd.merge(pivot, overall, on=["subject", "group"], how="left")

    merged["group_rank"] = np.where(merged["group"] == "Humans", 1, 0)
    merged = merged.sort_values(
        by=["group_rank", "overall_accuracy"],
        ascending=[True, False],
        kind="mergesort",
    )

    name_map = {
        "Hum-1": "Human-1",
        "Hum-2": "Human-2",
        "Hum-3": "Human-3",
        "Hum-4": "Human-4",
        "Doubao-1-5-pro-32k-250115": "Doubao-1-5-pro",
        "GPT4o-240806": "GPT-4o",
        "DeepSeek-v3_250324": "DeepSeek-v3",
        "Qwen2.5-Max-250409": "Qwen2.5-Max",
        "Doubao-1-5-lite-32k-250115": "Doubao-1-5-lite",
        "Qwen2.5-72B-Instruct": "Qwen2.5-72B",
        "GPT-4o-mini-2024-07-18": "GPT-4o-mini",
        "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
        "glm-4-9b-chat": "glm-4-9b",
        "Llama-3.1-8B-instruct": "Llama-3.1-8B",
        "internlm2_5-7b-chat": "internlm2_5-7b",
    }

    ordered_task_cols = [
        "SINGLE-T",
        "SINGLE-F",
        "BATCH-T",
        "BATCH-F",
        "SIM-GRA",
        "CAT-GRA",
        "CON-INS F*10",
        "CON-INS T5F5",
    ]
    ordered_task_cols = [t for t in ordered_task_cols if t in merged.columns]
    merged = merged[["subject", "group_rank"] + ordered_task_cols + ["overall_accuracy"]]
    merged = merged.rename(
        columns={
            "subject": "Model",
            "overall_accuracy": "Average",
        }
    )

    numeric_cols = [c for c in merged.columns if c not in ["Model", "group_rank"]]
    model_rows = merged[merged["group_rank"] == 0].drop(columns=["group_rank"])
    human_rows = merged[merged["group_rank"] == 1].drop(columns=["group_rank"])

    ave_llm = model_rows[numeric_cols].mean(numeric_only=True).to_frame().T
    ave_llm.insert(0, "Model", "Ave.LLM")
    ave_human = human_rows[numeric_cols].mean(numeric_only=True).to_frame().T
    ave_human.insert(0, "Model", "Ave.Human")

    merged = pd.concat([model_rows, ave_llm, human_rows, ave_human], ignore_index=True)

    merged["Model"] = merged["Model"].map(name_map).fillna(merged["Model"])
    merged[numeric_cols] = merged[numeric_cols].round(3)

    def format_value(val):
        if pd.isna(val):
            return ""
        val_str = f"{val:.3f}"
        if val_str.startswith("0."):
            return val_str[1:]
        if val_str.startswith("-0."):
            return "-" + val_str[2:]
        return val_str

    for col in numeric_cols:
        merged[col] = merged[col].apply(format_value)

    return merged


def main():
    print("\n1. Reading data...")
    df = pd.read_csv(EXCEL_FILE)

    # Build long format table
    print("2. Building Item x Subject Long Table...")
    df_long = build_long_df(df)
    print(f"   Total records: {len(df_long)}")

    # Output Subject x Task Accuracy Table (Sorted by Overall Average, Models Top, Humans Bottom)
    print("\n2.1 Outputting Subject x Task Accuracy Table...")
    subject_task_table = build_subject_task_accuracy_table(df_long)
    subject_task_file = os.path.join(OUTPUT_DIR, f"subject_task_accuracy_{TIMESTAMP}.csv")
    subject_task_table.to_csv(subject_task_file, index=False, encoding="utf-8-sig")
    print(f"  ✓ Subject-Task Accuracy Table saved: {subject_task_file}")

    # Draw Boxplot (Refer to boxplot_code.py)
    if PLOT_BOXPLOT:
        print("\n2.2 Drawing Boxplot...")
        df_plot = df_long[df_long["task"].isin(RUN_TASKS)].copy()
        if len(df_plot) == 0:
            print("  Warning: No valid data, skipping boxplot generation")
        else:
            # Calculate "member accuracy" by task x group member first
            acc_by_subject = (
                df_plot.groupby(["task", "group", "subject"], as_index=False)["score"]
                .mean()
                .rename(columns={"score": "accuracy"})
            )

            model_order = ["Humans", "larger-scale models", "semi-larger-scale models", "smaller-scale models"]
            task_order = RUN_TASKS
            color_map = {
                "Humans": "#dc267f",
                "larger-scale models": "#ffb000",
                "semi-larger-scale models": "#648fff",
                "smaller-scale models": "#785ef0",
            }

            # Position settings
            position_map = {}
            width = 0.15
            offsets = {
                "Humans": -1.5 * width,
                "larger-scale models": -0.5 * width,
                "semi-larger-scale models": 0.5 * width,
                "smaller-scale models": 1.5 * width,
            }
            for i, task in enumerate(task_order):
                base = i * 1.0
                for model in model_order:
                    pos = base + offsets[model]
                    position_map[(task, model)] = pos

            fig, ax = plt.subplots(figsize=(12, 6))
            legend_handles = {}
            for model in model_order:
                for task in task_order:
                    subset = acc_by_subject[(acc_by_subject["task"] == task) & (acc_by_subject["group"] == model)]
                    if len(subset) > 0:
                        bp = ax.boxplot(
                            subset["accuracy"],
                            positions=[position_map[(task, model)]],
                            widths=width,
                            patch_artist=True,
                            boxprops=dict(facecolor=color_map[model], color="black"),
                            medianprops=dict(color="black"),
                            showmeans=False,
                        )
                        if model not in legend_handles:
                            legend_handles[model] = bp["boxes"][0]

            ax.legend(
                legend_handles.values(),
                legend_handles.keys(),
                title="Model Group",
                fontsize=14,
                title_fontsize=14,
                loc="lower left",
            )
            ax.set_xticks([i for i in range(len(task_order))])
            ax.set_xticklabels(task_order, fontsize=14)
            ax.set_ylabel("Accuracy", fontsize=14)
            # Automatically adjust y-axis range
            if len(acc_by_subject) > 0:
                min_acc = acc_by_subject["accuracy"].min()
                max_acc = acc_by_subject["accuracy"].max()
                y_min = max(0.0, min_acc - 0.05)
                y_max = min(1.0, max_acc + 0.05)
                ax.set_ylim(y_min, y_max)
            else:
                ax.set_ylim(0.0, 1.0)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.tick_params(axis="y", labelsize=14)
            plt.tight_layout()

            boxplot_file = os.path.join(OUTPUT_DIR, f"boxplot_group_ranking_{TIMESTAMP}.pdf")
            plt.savefig(boxplot_file)
            print(f"  Boxplot saved to: {boxplot_file}")
            plt.close()

    # Modeling by Task
    print("\n3. Fitting Mixed Effects Models by Task...")
    stats_summary = []
    stats_adjacent = []
    for task in RUN_TASKS:
        print("\n" + "-" * 80)
        print(f"Task: {task}")
        df_task = df_long[df_long["task"] == task].copy()
        if task in ["BATCH", "CON-INS"]:
            summary, adjacent = run_lmm_for_task(df_task, task)
        else:
            summary, adjacent = run_glmm_for_task(df_task, task)
        if summary is not None:
            stats_summary.append(summary)
        if adjacent:
            for row in adjacent:
                row.update({"Task": task, "Model": summary["Model"] if summary else None})
            stats_adjacent.extend(adjacent)

    # Output CSV Summary Report
    summary_file = os.path.join(OUTPUT_DIR, f"mixed_model_summary_{TIMESTAMP}.csv")
    pd.DataFrame(stats_summary).to_csv(summary_file, index=False, encoding="utf-8-sig")
    print(f"  ✓ Statistical Summary CSV saved: {summary_file}")

    adjacent_file = os.path.join(OUTPUT_DIR, f"mixed_model_adjacent_{TIMESTAMP}.csv")
    pd.DataFrame(stats_adjacent).to_csv(adjacent_file, index=False, encoding="utf-8-sig")
    print(f"  ✓ Adjacent Test CSV saved: {adjacent_file}")

    # Output Score Table (Unified ID + Columns for each model score)
    print("\n4. Outputting Model Score Table...")
    score_table = build_score_table(df)
    score_table_file = os.path.join(OUTPUT_DIR, f"model_scores_{TIMESTAMP}.csv")
    score_table.to_csv(score_table_file, index=False, encoding="utf-8-sig")
    print(f"  ✓ Score Table saved: {score_table_file}")



if __name__ == "__main__":
    main()
