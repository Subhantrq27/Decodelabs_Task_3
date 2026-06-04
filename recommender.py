"""
╔══════════════════════════════════════════════════════════════════╗
║       DecodeLabs – AI Project 3: Tech Stack Recommender          ║
║       Content-Based Filtering via TF-IDF + Cosine Similarity     ║
╚══════════════════════════════════════════════════════════════════╝

Pipeline:
  Step 1 – Ingestion   : Capture user skill inputs
  Step 2 – Scoring     : Compute TF-IDF vectors + Cosine Similarity
  Step 3 – Sorting     : Rank items by similarity score (descending)
  Step 4 – Filtering   : Return Top-N results only
"""

import csv
import math
import os
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────

def load_dataset(filepath: str) -> list[dict]:
    """Load job roles and their skill tags from CSV."""
    dataset = []
    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            role   = row["job_role"].strip()
            skills = [s.strip().lower() for s in row["skills"].split()]
            dataset.append({"role": role, "skills": skills})
    return dataset


# ─────────────────────────────────────────────────────────────────
# STEP 1 – INGESTION: build user profile vector
# ─────────────────────────────────────────────────────────────────

def get_user_profile() -> list[str]:
    """
    Prompt the user for at least 3 skills/interests.
    Returns a normalised list of skill tokens.
    """
    print("\n" + "═"*60)
    print("  🤖  DecodeLabs Tech Stack Recommender")
    print("═"*60)
    print("\nEnter your skills / interests one at a time.")
    print("You need at least 3. Type 'done' when finished.\n")

    skills = []
    while True:
        entry = input(f"  Skill {len(skills)+1}: ").strip().lower()
        if entry == "done":
            if len(skills) < 3:
                remaining = 3 - len(skills)
                print(f"  ⚠  Please enter at least {remaining} more skill(s).")
            else:
                break
        elif entry == "":
            print("  ⚠  Input cannot be empty.")
        else:
            # replace spaces within a multi-word entry with underscores
            normalised = entry.replace(" ", "_")
            skills.append(normalised)
            print(f"  ✔  Added: {normalised}")

    print(f"\n  Profile captured → {skills}")
    return skills


# ─────────────────────────────────────────────────────────────────
# TF-IDF MACHINERY
# ─────────────────────────────────────────────────────────────────

def compute_tf(term_list: list[str]) -> dict[str, float]:
    """Term Frequency = count(term) / total_terms_in_document"""
    tf = defaultdict(float)
    total = len(term_list)
    for t in term_list:
        tf[t] += 1
    return {t: count / total for t, count in tf.items()}


def compute_idf(dataset: list[dict]) -> dict[str, float]:
    """
    Inverse Document Frequency = log(N / df(term))
    where df(term) = number of documents containing term.
    Uses the job-role skill lists as 'documents'.
    """
    N = len(dataset)
    df = defaultdict(int)
    for item in dataset:
        unique_skills = set(item["skills"])
        for skill in unique_skills:
            df[skill] += 1

    idf = {}
    for term, freq in df.items():
        idf[term] = math.log(N / freq)
    return idf


def build_tfidf_vector(term_list: list[str],
                       vocab: list[str],
                       idf: dict[str, float]) -> list[float]:
    """Convert a list of terms into a TF-IDF weighted vector over vocab."""
    tf = compute_tf(term_list)
    vector = []
    for term in vocab:
        tf_val  = tf.get(term, 0.0)
        idf_val = idf.get(term, 0.0)
        vector.append(tf_val * idf_val)
    return vector


def build_user_vector(user_skills: list[str],
                      vocab: list[str],
                      idf: dict[str, float]) -> list[float]:
    """
    User profile: each selected skill gets TF=1 (binary presence),
    weighted by IDF so rare skills carry more signal.
    """
    vector = []
    for term in vocab:
        idf_val = idf.get(term, 0.0)
        presence = 1.0 if term in user_skills else 0.0
        vector.append(presence * idf_val)
    return vector


# ─────────────────────────────────────────────────────────────────
# STEP 2 – SCORING: Cosine Similarity
# ─────────────────────────────────────────────────────────────────

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    cos(θ) = (A · B) / (||A|| × ||B||)
    Returns a value in [0, 1] because TF-IDF values are non-negative.
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a ** 2 for a in vec_a))
    magnitude_b = math.sqrt(sum(b ** 2 for b in vec_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0          # Cold-start guard
    return dot_product / (magnitude_a * magnitude_b)


# ─────────────────────────────────────────────────────────────────
# STEPS 3 & 4 – SORTING + FILTERING
# ─────────────────────────────────────────────────────────────────

def rank_and_filter(scored_items: list[tuple[str, float]],
                    top_n: int = 3) -> list[tuple[str, float]]:
    """Sort by score descending, return Top-N items only."""
    sorted_items = sorted(scored_items, key=lambda x: x[1], reverse=True)
    return sorted_items[:top_n]


# ─────────────────────────────────────────────────────────────────
# OUTPUT DISPLAY
# ─────────────────────────────────────────────────────────────────

def display_results(top_roles: list[tuple[str, float]],
                    dataset: list[dict],
                    user_skills: list[str]) -> None:
    """Pretty-print the Top-N recommendations."""
    print("\n" + "═"*60)
    print("  🏆  TOP CAREER RECOMMENDATIONS FOR YOU")
    print("═"*60)

    medals = ["🥇", "🥈", "🥉"]

    for rank, (role, score) in enumerate(top_roles, start=1):
        medal = medals[rank - 1] if rank <= 3 else f"#{rank}"
        bar_len = int(score * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)

        # find matching skills for explanation
        role_skills = next(
            (item["skills"] for item in dataset if item["role"] == role), []
        )
        matched = [s for s in user_skills if s in role_skills]

        print(f"\n  {medal}  {role}")
        print(f"      Match Score : {score:.4f}  [{bar}]")
        print(f"      Match %     : {score*100:.1f}%")
        if matched:
            print(f"      Shared Tags : {', '.join(matched)}")
        else:
            print(f"      Shared Tags : (indirect match via vector alignment)")

    print("\n" + "═"*60)
    print("  ✅  Recommendation complete.")
    print("═"*60 + "\n")


# ─────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────

def run_recommender(csv_path: str = "raw_skills.csv", top_n: int = 3) -> None:
    # ── Load data ──────────────────────────────────────────────────
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    dataset = load_dataset(csv_path)
    print(f"\n  📂  Loaded {len(dataset)} job roles from '{csv_path}'")

    # ── Build shared vocabulary ────────────────────────────────────
    vocab = sorted({skill for item in dataset for skill in item["skills"]})
    print(f"  📖  Vocabulary size: {len(vocab)} unique skills")

    # ── Compute IDF over the entire dataset ────────────────────────
    idf = compute_idf(dataset)

    # ── STEP 1: Ingestion ──────────────────────────────────────────
    user_skills = get_user_profile()

    # Add unseen user skills to IDF with a high weight (rare = specific)
    for skill in user_skills:
        if skill not in idf:
            idf[skill] = math.log(len(dataset))   # treat as appearing once
        if skill not in vocab:
            vocab.append(skill)

    vocab = sorted(set(vocab))  # keep vocab consistent

    # ── STEP 2: Scoring ────────────────────────────────────────────
    user_vec = build_user_vector(user_skills, vocab, idf)

    scored_items = []
    for item in dataset:
        item_vec = build_tfidf_vector(item["skills"], vocab, idf)
        score    = cosine_similarity(user_vec, item_vec)
        scored_items.append((item["role"], score))

    # ── STEPS 3 & 4: Sort + Filter ─────────────────────────────────
    top_roles = rank_and_filter(scored_items, top_n=top_n)

    # ── Display ────────────────────────────────────────────────────
    display_results(top_roles, dataset, user_skills)

    # ── Ask to try again ──────────────────────────────────────────
    again = input("  Try with different skills? (yes/no): ").strip().lower()
    if again in ("yes", "y"):
        run_recommender(csv_path, top_n)


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    CSV_PATH = os.path.join(os.path.dirname(__file__), "raw_skills.csv")
    run_recommender(csv_path=CSV_PATH, top_n=3)
