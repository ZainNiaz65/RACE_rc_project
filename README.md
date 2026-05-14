--> Intelligent Reading Comprehension and Quiz Generation System

An AI-powered system that automatically generates comprehension questions, verifies answers, creates distractors, and provides hints using Traditional Machine Learning (No Neural Networks).

--> Project Overview

This system builds on the RACE dataset (Reading Comprehension from Examinations) to create an intelligent reading comprehension assistant. The system integrates two specialized ML pipelines exposed through an interactive web interface.

--> What the System Does

- Answer Verification (Model A): Predicts whether a selected answer option is correct for a given question
- Quiz Generation: Creates multiple-choice questions from reading passages
- Distractor Generation (Model B): Produces plausible but incorrect answer options
- Hint Generation: Provides graduated hints to guide users toward correct answers

--> Technology Stack

| Component              | Technology                                                                   |
|------------------------|------------------------------------------------------------------------------|
| Core ML                | scikit-learn (Logistic Regression, SVM, Random Forest, XGBoost, Naive Bayes) |
| Text Features          | TF-IDF Vectorization, One-Hot Encoding                                       |
| Unsupervised Learning  | K-Means Clustering                                                           |
| Semi-Supervised        | Label Propagation                                                            |
| Word Embeddings        | Word2Vec (Gensim)                                                            |
| UI Framework           | Streamlit                                                                    |
| Data Processing        | Pandas, NumPy                                                                |

--> Project Structure

race_rc_project/
├── data/
│ └── raw/
│ ├── train.csv # RACE training data
│ ├── dev.csv # RACE validation data
│ └── test.csv # RACE test data
├── models/
│ ├── model_a/
│ │ └── traditional/
│ │ ├── logistic_regression.pkl
│ │ ├── svm.pkl
│ │ ├── random_forest.pkl
│ │ ├── xgboost.pkl
│ │ ├── ensemble_voting.pkl
│ │ └── preprocessor.pkl
│ └── model_b/
│ └── traditional/
│ ├── distractor_ranker.pkl
│ ├── hint_scorer.pkl
│ ├── word2vec.model
│ └── vocabulary.pkl
├── src/
│ ├── preprocessing.py # Data cleaning & feature engineering
│ ├── model_a_train.py # Answer verification training
│ ├── model_b_train.py # Distractor & hint training
│ ├── inference.py # Unified prediction API
│ └── evaluate.py # Model evaluation metrics
├── ui/
│ └── app.py # Streamlit web interface
├── notebooks/
│ ├── EDA.ipynb # Exploratory data analysis
│ └── experiments.ipynb # Experiment tracking
├── requirements.txt
└── README.md


--> Installation

mkdir i230700-i230077-G
cd i230700-i230077-G

``` virtual environment
python -m venv venv   ( Windows )
venv\Scripts\activate

``` install dependencies
pip install -r requirements.txt

``` download RACE dataset
Download the RACE dataset from Kaggle:
                                        https://www.kaggle.com/datasets/racecsv/race

Place the files in data/raw/:
                            train.csv
                            dev.csv
                            test.csv

``` Train Model A (Answer Verification)
python src/model_a_train.py

``` Train Model B (Distractor & Hint Generator)
python src/model_b_train.py

``` Running the Application and Start the Streamlit UI
streamlit run ui/app.py

``` Sample Test Input
Quiz_data.txt

