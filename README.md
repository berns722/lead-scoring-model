# Lead Scoring Model

A machine learning pipeline that scores leads by their probability of converting into paying customers, helping sales teams prioritize outreach.

**🔗 Live demo: [lead-scoring-model.streamlit.app](https://lead-scoring-model.streamlit.app/)**

## Business Problem

EdTech platforms like ExtraaLearn generate large volumes of leads, but only a fraction convert into paying customers. Identifying high-probability leads early lets sales teams allocate effort efficiently. This project builds a lead scoring model on behavioral and demographic data, evaluates it as both a classifier and a ranking tool, and surfaces the factors that drive conversion.

## Dataset

- **4,612** leads, 14 features capturing demographics, platform behavior, and marketing touchpoints
- Target: `status` (1 = converted, 0 = not converted)
- Class imbalance: ~70% non-converted, ~30% converted

## Project Structure

```
lead-scoring-model/
├── src/
│   ├── data_loader.py
│   ├── data_preprocessing.py
│   ├── eda.py
│   └── evaluate.py
├── notebooks/
│   └── exploration.ipynb
├── app/
│   ├── streamlit_app.py
│   ├── requirements.txt
│   └── models/
│       └── learn_model.joblib
├── data/
│   └── ExtraaLearn.csv
├── environment.yml
└── README.md
```

## Approach

- **Models compared:** Decision Tree (interpretable baseline), Random Forest (bagging), Gradient Boosting (boosting) — all three share the same preprocessing pipeline for a fair comparison
- **Preprocessing:** `ColumnTransformer` one-hot encodes categorical features and passes numerical features through unchanged; tree-based models are scale-invariant, so no scaling is applied
- **Tuning:** `GridSearchCV` on the winning architecture, scored on recall with 5-fold cross-validation. Tuning explored `n_estimators`, `max_depth`, and `learning_rate` but found defaults already well-regularized — the un-tuned model was retained
- **Imbalance:** no resampling or class weighting applied; recall-focused evaluation addresses the business priority directly

## Evaluation

Recall is the primary metric — the business cost of missing a converter outweighs the cost of a false positive. F1, precision, and accuracy are tracked alongside.

| Metric | Decision Tree | Random Forest | Gradient Boosting |
|---|---|---|---|
| Recall | 65.9% | 68.5% | **72.0%** |
| F1 | 65.9% | 73.4% | **75.3%** |
| Precision | 66.0% | 79.2% | **79.0%** |
| Accuracy | 79.3% | 84.9% | **85.6%** |

Gradient Boosting wins on every test metric. It also produces the smallest train/test gap (~5 points vs ~30 for the other two), reflecting boosting's inherent regularization through learning-rate-controlled sequential trees. The default Gradient Boosting is serialized as the final model.

## Ranking Analysis

A classifier's threshold metrics tell only part of the story. The model's true value as a scoring tool is captured by how well it *ranks* leads — sales can call the top-K highest-scored leads without committing to a binary cutoff.

| Top K leads | Converters in top K | Precision@K | Share of total converters reached |
|---|---|---|---|
| 10 | 10 / 10 | 100% | 2.4% |
| 50 | 50 / 50 | 100% | 11.8% |
| 100 | 93 / 100 | 93.0% | 22.0% |
| 200 | 177 / 200 | 88.5% | 41.9% |

The first 50 ranked leads are all converters — a 100% hit rate at the top of the list. Even at K=200 (14% of the test set), precision remains at 88.5% while reaching 42% of all converters. Compared to random calling (~30% conversion rate at any position), this is a ~3× lift at the top of the list.

## Key Drivers of Conversion

| Rank | Feature | Importance |
|---|---|---|
| 1 | Time spent on website | 0.27 |
| 2 | First interaction: Website | 0.21 |
| 3 | Profile completion: High | 0.16 |
| 4 | First interaction: Mobile App | 0.12 |

Time spent on website is the strongest predictor, consistent with the correlation analysis. The model surfaces first-interaction channel in both directions — website-first leads convert at a much higher rate, mobile-first leads convert substantially less. Marketing channel variables (print, digital media, referrals) contribute negligibly, suggesting either limited reach to converting leads or limited capture in the dataset.

## How to Run

**Live demo:** [lead-scoring-model.streamlit.app](https://lead-scoring-model.streamlit.app/)

Try the interactive scorer directly — no installation needed. To run locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/berns722/lead-scoring-model.git
   cd lead-scoring-model
   ```
2. Create and activate the environment:
   ```bash
   conda env create -f environment.yml
   conda activate lead-scoring-model
   ```
3. Launch the notebook:
   ```bash
   jupyter notebook notebooks/exploration.ipynb
   ```
4. Run all cells to reproduce the analysis and re-serialize the model to `app/models/learn_model.joblib`.

## Future Directions

- Lead segmentation: cluster leads into personas using the same feature set
- Probability calibration: check whether predicted probabilities match real conversion rates
- Statistical significance testing across models (McNemar, 5x2cv) for model selection rigor
- Model monitoring and periodic retraining as new conversion data accumulates

