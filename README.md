# Mental Health Sentiment Analysis

Understanding patterns, emotions, and topics in large-scale online mental-health conversations helps researchers, clinicians, and platform owners anticipate emerging issues and tailor support resources. This repository contains an **end-to-end, fully reproducible pipeline** that cleans a 53 k-post corpus, performs rule-based and machine-learning sentiment analysis, extracts latent topics, and visualises statistically significant insights.

---
## 🎯 Project Aim
The project seeks to **quantify emotions, uncover prevalent topics, and analyse their statistical relationships** in user-generated text about mental health. By transforming noisy social‑media posts into structured evidence, we aim to answer:
1. *What proportion of the discourse is positive, negative, or mixed?*
2. *Which themes (e.g., depression, anxiety, self-harm) dominate and how do they evolve with sentiment?*
3. *Are topic–sentiment interactions statistically significant, and can they inform early‑warning systems or support interventions?*

---
## 🔬 Methodology
| Stage | Description |
| ----- | ----------- |
| **Data ingestion** | 53 043 raw posts (Reddit‑style data dump) with seven author‑provided status labels. |
| **Cleaning & filtering** | Remove duplicates, URLs, emojis; lowercase; drop non‑English or null text → 20 461 unique mental‑health posts. |
| **Pre‑processing** | *NLTK* tokenisation, custom stop‑word removal, lemmatisation, length outlier trimming. |
| **Sentiment mapping** | Collapse 7 raw labels → `negative`, `neutral`, `mixed`; compute VADER compound scores and validate (66 % accuracy). |
| **Topic modelling** | TF‑IDF vectorisation + LDA *(k = 5)*; heuristic label assignment (e.g., *“suicidal ideation”*, *“social isolation”*). |
| **Statistical tests** | χ² test for independence (topics × sentiment), Welch *t*-test on mean VADER scores, OLS regression to quantify topic predictors (*R² = 0.18*). |
| **Visual analytics** | Matplotlib/seaborn/Plotly histograms, bar plots, word clouds; interactive Jupyter notebook. |

---
## 📈 Key Findings
* **Sentiment distribution** – 58 % negative, 31 % neutral, 11 % mixed.  
* **Dominant topics** – *Depression* (23 %), *Anxiety* (19 %), *Suicidal ideation* (12 %), *Medication & therapy* (9 %), *Social support & relationships* (8 %).  
* **Significant association** – χ² = 140.3, *p* < 0.001: negative sentiment is disproportionately linked to *Depression* & *Suicidal ideation*, whereas neutral posts skew toward *Medication* discussions.  
* **Regression insights** – Variables `has_depression`, `has_suicide`, `has_stress` have negative beta coefficients; `emotion_happy` and `emotion_neutral` have positive ones (*R² = 0.18*).

Full statistics are archived in `docs/model_summary.txt`, and corresponding plots live in `figures/`.

---
## 📝 Conclusion
The analysis confirms that online mental‑health forums are dominated by negative emotions encompassing depression and self‑harm ideation. However, a substantial portion of neutral discussion focuses on treatment strategies, indicating the platform’s dual role as both a distress outlet and a peer‑support hub. Quantified topic–sentiment links can guide **early‑warning dashboards** for moderators and inform **targeted resource recommendations** for vulnerable users.

Future work may incorporate transformer‑based sentiment models, temporal trend detection, and real‑time streaming architectures.

---
## ✨ Key Features
- **Data cleaning & de‑duplication** – removes noise, URLs, emojis and duplicate posts; filters for ~20 k mental‑health related entries.
- **Text pre‑processing** – tokenisation (*NLTK*), custom stop‑word removal, lemmatisation, length analysis, outlier handling.
- **Sentiment & emotion scoring** – maps 7 raw status labels → `negative/neutral/mixed` and computes VADER compound scores with 66 % validation accuracy.
- **Topic modelling** – TF‑IDF + LDA (5 topics) with auto‑generated human‑readable labels and top‑word bar charts.
- **Statistical testing** – χ² tests, independent *t*-test, and linear regression (*R² = 0.18*) to quantify relationships between topics, keywords and sentiment.
- **Rich visual analytics** – keyword co‑occurrence heat‑map, sentiment/emotion distributions, word clouds, and more (see `figures/`).
- **Reproducible notebook** – `Mental Health Sentiment Analysis.ipynb` executes the full workflow in one click.

---
## 📂 Repository structure
```text
mental-health-sentiment-analysis/
├── data/                     # Combined Data.csv (raw) + posts_sentiment.csv (processed)
├── docs/
│   ├── model_summary.txt     # Regression & test statistics
│   └── impact_analysis.md    # Narrative interpretation of results
├── figures/                  # All generated PNG figures
├── Mental Health Sentiment Analysis.ipynb
├── requirements.txt          # Locked package versions
└── README.md                 # ⇽ you are here
```

---
## 📚 Dataset
| Metric | Value |
| ------ | ----- |
| Raw posts | **53 043** |
| Labels | Normal, Depression, Suicidal, Anxiety, Bipolar, Stress, Personality disorder |
| Null rows dropped | 362 (< 1 %) |
| Filtered for mental‑health keywords | 22 256 |
| Unique, cleaned posts | **20 461** |

---
## 🔧 Installation
```bash
# 1. Clone and enter repo
git clone https://github.com/<your-handle>/mental-health-sentiment-analysis.git
cd mental-health-sentiment-analysis

# 2. Create and activate virtualenv
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the notebook
jupyter notebook "Mental Health Sentiment Analysis.ipynb"
```
> **Python ≥ 3.9** is recommended. GPU is **not** required; the heaviest model is VADER (rule‑based).

**requirements.txt**
```
pandas>=2.0.0
numpy>=1.26
nltk>=3.8
scikit-learn>=1.4
matplotlib>=3.8
seaborn>=0.13
plotly>=5.20
wordcloud>=1.9
transformers>=4.52
torch>=2.6
safetensors>=0.5
```

---
## 🚀 Quick Start
Run the whole pipeline from the notebook *or* headless via the optional script:
```bash
python scripts/run_pipeline.py \
  --input data/Combined\ Data.csv \
  --output-dir .
```
Outputs (clean CSVs, figures, model summaries) will be saved in `data/`, `figures/`, and `docs/` respectively.

---
## 🤝 Contributing
Pull requests are welcome! Please open an issue first to discuss proposed changes.

---
## 📜 License
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---
## 🙏 Acknowledgements
- **VADER Sentiment** – Hutto & Gilbert (2014)
- **NLTK**, **scikit-learn**, **HuggingFace Transformers**
- Inspiration from mental‑health text‑mining research communities.

---
## ✉️ Contact
Maintainer — **Shubham Goel**  
Feel free to reach out via LinkedIn or raise an issue.
