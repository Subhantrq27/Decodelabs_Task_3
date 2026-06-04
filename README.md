# Tech Stack Recommender — AI Project 3

> **DecodeLabs Industrial Training Kit · Batch 2026**  
> Content-Based Filtering using TF-IDF Weighting + Cosine Similarity

---

## Overview

A command-line AI recommendation engine that maps a user's raw skills and career interests to the most relevant tech job roles. Built from scratch using **pure Python** — no ML libraries — to demonstrate mastery of the core mathematics behind recommendation systems.

Enter 3+ skills and get your Top 3 career matches, ranked by similarity score.

---

## How It Works

The system follows a strict **Input → Process → Output (IPO)** pipeline:

```
User Skills Input
      │
      ▼
 [TF-IDF Vectorization]   ← weights rare/specific skills higher
      │
      ▼
 [Cosine Similarity]       ← measures angular alignment between vectors
      │
      ▼
 [Sort + Top-N Filter]     ← returns only highest-scoring matches
      │
      ▼
Ranked Job Role Recommendations
```

### Why TF-IDF?
Simple binary matching treats "Python" and "Transformers" equally. TF-IDF penalises common skills and rewards rare, specific ones — so `kubernetes` carries more signal than `python` in differentiation.

### Why Cosine Similarity?
Euclidean distance is sensitive to vector magnitude (a long job description vs a short one would appear "far apart" even if they share the same topics). Cosine similarity measures **direction**, not size — making it magnitude-invariant and ideal for text-based feature spaces.

---

## Project Structure

```
tech-stack-recommender/
├── recommender.py      # Main pipeline (zero external dependencies)
├── raw_skills.csv      # Dataset: 22 job roles x their skill tags
└── README.md
```

---

## Pipeline Steps

| Step | Name | Description |
|------|------|-------------|
| 1 | **Ingestion** | Accepts 3+ user skill inputs; normalises to lowercase tokens |
| 2 | **Scoring** | Builds TF-IDF vectors for all roles + user; computes Cosine Similarity |
| 3 | **Sorting** | Ranks all 22 job roles by similarity score (descending) |
| 4 | **Filtering** | Returns Top-3 results to prevent choice overload |

---

## Getting Started

**Requirements:** Python 3.10+ · No external libraries needed

```bash
# Clone the repo
git clone https://github.com/<your-username>/tech-stack-recommender.git
cd tech-stack-recommender

# Run
python recommender.py
```

**Example session:**

```
  Loaded 22 job roles from 'raw_skills.csv'
  Vocabulary size: 115 unique skills

  DecodeLabs Tech Stack Recommender

  Enter your skills / interests one at a time.
  You need at least 3. Type 'done' when finished.

  Skill 1: python
  Added: python
  Skill 2: machine learning
  Added: machine_learning
  Skill 3: tensorflow
  Added: tensorflow
  Skill 4: done

  TOP CAREER RECOMMENDATIONS FOR YOU

  #1  ML Engineer
      Match Score : 0.3115  [████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
      Match %     : 31.2%
      Shared Tags : machine_learning, tensorflow

  #2  Data Scientist
      Match Score : 0.2471  [█████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
      Match %     : 24.7%
      Shared Tags : machine_learning, tensorflow

  #3  NLP Engineer
      Match Score : 0.2238  [████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
      Match %     : 22.4%
      Shared Tags : machine_learning, tensorflow
```

---

## Dataset

`raw_skills.csv` contains **22 job roles** across domains:

| Domain | Roles Included |
|--------|----------------|
| AI / ML | Data Scientist, ML Engineer, AI Research Scientist, NLP Engineer, Computer Vision Engineer |
| Cloud / DevOps | DevOps Engineer, Cloud Architect, Site Reliability Engineer |
| Development | Backend Developer, Frontend Developer, Full Stack Developer, Mobile Developer |
| Data | Data Engineer, Data Analyst, Database Administrator, Quantitative Analyst |
| Specialised | Cybersecurity Analyst, Blockchain Developer, Robotics Engineer, Game Developer, Systems Programmer, Product Manager |

You can extend the dataset by adding rows to `raw_skills.csv` — no code changes needed.

---

## Core Math

**Term Frequency:**
```
TF(t, d) = count(t in d) / total terms in d
```

**Inverse Document Frequency:**
```
IDF(t) = log( N / df(t) )
```

**TF-IDF Weight:**
```
w(t, d) = TF(t, d) x IDF(t)
```

**Cosine Similarity:**
```
cos(theta) = (A . B) / (||A|| x ||B||)
```
Score of `1.0` = perfect alignment · `0.0` = no shared characteristics

---

## Cold Start Handling

- **Item Cold Start** — not an issue; new roles are immediately scoreable from their metadata tags.
- **User Cold Start** — guarded by enforcing a minimum of 3 skill inputs before scoring begins.

---

## Built With

- **Python 3.10+** — standard library only (`csv`, `math`, `os`, `collections`)
- **Algorithm:** Content-Based Filtering · TF-IDF · Cosine Similarity
- **Project:** DecodeLabs AI Industrial Training — Project 3

---

*Part of the DecodeLabs Batch 2026 AI Engineering track.*
