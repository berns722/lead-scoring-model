# Lead Scoring Model

A machine learning pipeline that predicts whether a lead will convert into a paying customer, helping sales teams prioritize outreach and improve conversion rates.

## Business Problem

EdTech platforms like ExtraaLearn generate large volumes of leads, but only a fraction convert into paying customers. Identifying high-probability leads early lets sales teams allocate effort efficiently. This project builds an interpretable classification model to predict lead conversion from behavioral and demographic data, and surfaces the factors that drive it.

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
│   ├── model.py
│   └── evaluate.py
├── notebooks/
│   └── exploration.ipynb
├── data/
│   └── ExtraaLearn.csv
├── app/
└── README.md
```

## Approach

- **Model:** Decision Tree Classifier — an interpretable baseline whose splits map directly to actionable rules
- **Preprocessing:** `ColumnTransformer` one-hot encodes categorical features and passes numerical features through; trees are scale-invariant, so no scaling is applied
- **Tuning:** `GridSearchCV` over `max_depth`, `max_leaf_nodes`, and `min_samples_leaf`, scored on recall with 5-fold cross-validation
- **Imbalance:** no resampling or class weighting applied; recall-focused tuning addresses the business priority directly

## Evaluation

Recall is the priority metric — the business cost of missing a converter outweighs the cost of a false positive. F1, precision, and accuracy are tracked alongside it.

| Metric | Baseline | Tuned |
|---|---|---|
| Recall | 65.9% | **69.7%** |
| F1 | 65.9% | **74.9%** |
| Precision | 66.0% | **81.0%** |
| Accuracy | 79.2% | **85.8%** |

The unconstrained baseline overfits (perfect training scores, lower test scores). Tuning improves every test metric and narrows the train/test gap. The tuned model is serialized for deployment.

**Strongest predictors of conversion:** time spent on website, followed by website-first interaction and high profile completion — consistent across the correlation analysis and the model's feature importances.

## How to Run

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
4. Run all cells to retrain and serialize the model to `backend_files/learn_model.joblib`.

## Future Directions

- Deploy the model via the Streamlit app in `app/`
- Compare against ensemble methods (Random Forest, Gradient Boosting)
- Deeper segmentation of high-confidence predicted converters
- Model monitoring and feature engineering
- Probability ranking

