from Orange.data import Table, Domain, StringVariable
from Orange.preprocess.text import preprocess_strings
from Orange.classification import NaiveBayesLearner
import pickle
import pandas as pd

# ----------------------------
# LOAD DATASET
# ----------------------------
data = pd.read_csv("news.csv")   # Columns: text, label

# Define Orange domain
domain = Domain([StringVariable("text")], class_vars=StringVariable("label"))

# Convert to Orange Table
rows = [[row['text'], row['label']] for _, row in data.iterrows()]
orange_table = Table.from_list(domain, rows)

# Preprocess
orange_table.X = preprocess_strings(orange_table.X)

# ----------------------------
# TRAIN MODEL
# ----------------------------
learner = NaiveBayesLearner()
model = learner(orange_table)

# ----------------------------
# SAVE MODEL
# ----------------------------
with open("orange_news_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("🎉 ORANGE MODEL TRAINED & SAVED AS orange_news_model.pkl")
