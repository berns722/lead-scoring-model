# Lead Conversion Prediction System

A machine learning pipeline that predicts whether a lead will convert into a paying customer, enabling more efficient sales targeting.


## Business Problem

Companies often generate large volumes of leads, but only a small percentage convert into paying customers. Identifying high-quality leads early allows sales teams to prioritize outreach and improve conversion reates.

This project buildas a classification model to predict lead conversion using behavioral and demographic data.

## Dataset Overview

- Number of observations: **4612**
- Features:
- Target: Conversion (0 = No, 1 = Yes)
- Class imbalance: ~70% non-converted, ~30% converted

## Project Structure
```
lead-conversion/
│
├── src/
│   ├── data_loader.py
│   ├── data_preprocessing.py
│   ├── model.py
│   └── evaluate.py
│
├── notebooks/
│   └── exploration.ipynb
│
├── data/
│   └── ExtraaLearn.csv
│
├── app/
│
└── README.md
```
## How to Run

1. Clone the repository:
```bash
   git clone https://github.com/berns722/lead-scoring-model.git
   cd lead-scoring-model
```

2. Create and activate the conda environment:
```bash
   conda env create -f environment.yml
   conda activate ml-lab
```

3. Launch the notebook:
```bash
   jupyter notebook notebooks/exploration.ipynb
```

4. To retrain and serialize the model, run all cells in the notebook.
   The trained model will be saved to `backend_files/learn_model.joblib`.

5. *(Coming soon)* Launch the Streamlit app:
```bash
   streamlit run app/main.py
```
## ML Pipeline

1. Data loading
2. Data preprocessing
3. Train-test split
4. Model training (Decision Tree)
5. Evaluation using classification metrics

## Model & Approach

- Model: Decision Tree Classifier
- Reason: Interpretable baseline model
- Hyperparameter tuning: GridSearchCV
- Handling imbalance: (mention if using class_weight or not)

## Evaluation Metrics

The model is evaluated using:

- Recall (priority): to capture as many potential converters as possible
- Precision
- F1 Score
- Accuracy

Recall and F1 are prioritized due to the business objective of minimizing missed conversion opportunities.

## Results

**Baseline Model:**
 
 - Recall: 58%
 - F1 Score: 62.1%

 **Tuned Model:**

  - Recall: 58.3%
   - F1 Score: 62.6%

 Marginal improvement over baseline suggests the decision tree 
is near its performance ceiling on this dataset. 
Ensemble methods (Random Forest, Gradient Boosting) are 
identified as the logical next step.

## How to Run

 1. Clone the repository: ```git clone https://github.com/berns722/lead-conversion-system```
 2. Navigate to the project: ```cd lead-conversion```
 3. Install dependencies: ```pip install -r requirements.txt```
 4. Run the notebook or script: jupyter notebook

## Key Learnings

- Importans of aligning ML metrics with business objectives
- Trade-offs between recall and precision in classification problems
- Value of modularizing code for reproducibility
- Challenges of handling categorical data in pipelines

## Future Improvements

- Deploy model using streamlit
- Experiment with ensemble models (Random Forest, Gradient Boosting)
- Add model monitoring
- Improve feature engineering
 

