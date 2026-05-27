import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.metrics import accuracy_score, cohen_kappa_score
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM


HUMAN_COLS = ["H0", "H1", "H2"]
AI_COLS = ["AI1", "AI2", "AI3"]


def get_h0(row):
    a = str(row["A"]).strip().upper()
    b = str(row["B"]).strip().upper()
    if a == "X":
        return "A"
    if b == "X":
        return "B"
    raise ValueError(f"No X found for row {row.name}")


def load_ratings(file_path):
    df = pd.read_excel(file_path, engine="odf")
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed")]

    # Derive H0 from A/B marker columns
    df["H0"] = df.apply(get_h0, axis=1)

    needed_cols = HUMAN_COLS + AI_COLS
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    ratings = df[needed_cols].copy()
    ratings = ratings.apply(lambda s: s.astype(str).str.strip().str.upper())

    item_id = df["Pair"] if "Pair" in df.columns else pd.Series(range(len(df)), index=df.index)
    return df, ratings, item_id


def fleiss_from_columns(df_raters):
    table, _ = aggregate_raters(df_raters.to_numpy())
    return fleiss_kappa(table)


def majority_vote(row):
    # For 3 binary raters, this should always return a unique majority
    return row.mode().iloc[0]


def build_label_long_df(ratings, item_id, human_cols, ai_cols):
    frames = []

    for col in human_cols:
        frames.append(
            pd.DataFrame({
                "item_id": item_id,
                "rater_id": col,
                "group": "Human",
                "label": ratings[col],
            })
        )

    for col in ai_cols:
        frames.append(
            pd.DataFrame({
                "item_id": item_id,
                "rater_id": col,
                "group": "AI",
                "label": ratings[col],
            })
        )

    long_df = pd.concat(frames, ignore_index=True)
    long_df["label_B"] = (long_df["label"] == "B").astype(int)
    long_df["group_AI"] = (long_df["group"] == "AI").astype(int)
    return long_df


def fit_gee(formula, data):
    return smf.gee(
        formula,
        groups="item_id",
        data=data,
        family=sm.families.Binomial(),
    ).fit()


def fit_glmm(formula, data):
    return BinomialBayesMixedGLM.from_formula(
        formula,
        {"item": "0 + C(item_id)"},
        data,
    ).fit_vb()


file_path = "rater/ImageAnnotation_ai_aware_captioning_condition.ods"

df, ratings, item_id = load_ratings(file_path)

# Majority labels
ratings["HumanMajority"] = ratings[HUMAN_COLS].apply(majority_vote, axis=1)
ratings["AIMajority"] = ratings[AI_COLS].apply(majority_vote, axis=1)

# Fleiss' kappas
fleiss_kappa_humans = fleiss_from_columns(ratings[HUMAN_COLS])
fleiss_kappa_ai = fleiss_from_columns(ratings[AI_COLS])

# Cohen's kappa: human majority vs AI majority
cohen_human_vs_ai_majority = cohen_kappa_score(
    ratings["HumanMajority"],
    ratings["AIMajority"]
)

# Accuracy: only here H0 is treated as ground truth
acc_h1 = accuracy_score(ratings["H0"], ratings["H1"])
acc_h2 = accuracy_score(ratings["H0"], ratings["H2"])

acc_ai1 = accuracy_score(ratings["H0"], ratings["AI1"])
acc_ai2 = accuracy_score(ratings["H0"], ratings["AI2"])
acc_ai3 = accuracy_score(ratings["H0"], ratings["AI3"])
acc_ai_majority = accuracy_score(ratings["H0"], ratings["AIMajority"])

n_h1 = accuracy_score(ratings["H0"], ratings["H1"], normalize=False)
n_h2 = accuracy_score(ratings["H0"], ratings["H2"], normalize=False)

n_ai1 = accuracy_score(ratings["H0"], ratings["AI1"], normalize=False)
n_ai2 = accuracy_score(ratings["H0"], ratings["AI2"], normalize=False)
n_ai3 = accuracy_score(ratings["H0"], ratings["AI3"], normalize=False)
n_ai_majority = accuracy_score(ratings["H0"], ratings["AIMajority"], normalize=False)

n_total = len(ratings)

# All-are-raters model: no ground truth
label_long_df = build_label_long_df(ratings, item_id, HUMAN_COLS, AI_COLS)

gee_label_result = fit_gee("label_B ~ group_AI", label_long_df)
glmm_label_result = fit_glmm("label_B ~ group_AI", label_long_df)

print("Fleiss' kappa among human raters (H0, H1, H2):", fleiss_kappa_humans)
print("Fleiss' kappa among AI raters (AI1, AI2, AI3):", fleiss_kappa_ai)
print("Cohen's kappa between human majority and AI majority:", cohen_human_vs_ai_majority)
print()

print(f"H1 accuracy vs H0: {acc_h1:.2%} ({n_h1}/{n_total})")
print(f"H2 accuracy vs H0: {acc_h2:.2%} ({n_h2}/{n_total})")
print()

print(f"AI1 accuracy vs H0: {acc_ai1:.2%} ({n_ai1}/{n_total})")
print(f"AI2 accuracy vs H0: {acc_ai2:.2%} ({n_ai2}/{n_total})")
print(f"AI3 accuracy vs H0: {acc_ai3:.2%} ({n_ai3}/{n_total})")
print(f"AI majority accuracy vs H0: {acc_ai_majority:.2%} ({n_ai_majority}/{n_total})")
print()

print("All-are-raters model: label_B ~ group_AI")
print(gee_label_result.summary())
print("GEE odds ratio (AI vs Human):", float(np.exp(gee_label_result.params["group_AI"])))
print("GEE p-value:", gee_label_result.pvalues["group_AI"])
print()

print(glmm_label_result.summary())
print("GLMM odds ratio (AI vs Human):", float(np.exp(glmm_label_result.params[1])))