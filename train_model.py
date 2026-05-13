import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor

# =====================================
# LOAD DATASET
# =====================================

df = pd.read_csv("cleaned_data.csv")

# Remove missing values
df = df.dropna()

# =====================================
# LOCATION ENCODING
# =====================================

location_dummies = pd.get_dummies(df['location'])

df = pd.concat([df, location_dummies], axis=1)

df = df.drop('location', axis=1)

# =====================================
# FEATURES & TARGET
# =====================================

X = df.drop('price', axis=1)

y = df['price']

# =====================================
# TRAIN MODEL
# =====================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# =====================================
# SAVE MODEL
# =====================================

pickle.dump(
    model,
    open("house_price_model.pkl", "wb")
)

print("Advanced model trained successfully")

accuracy = model.score(X, y)

print("Model Accuracy:", accuracy)

# =====================================
# GRAPH 1
# HOUSE PRICE VS TOTAL SQFT
# =====================================

plt.figure(figsize=(12,7))

sns.scatterplot(
    x=df['total_sqft'],
    y=df['price'],
    color='blue',
    alpha=0.6
)

plt.title(
    "House Price vs Total Square Feet",
    fontsize=18,
    fontweight='bold'
)

plt.xlabel(
    "Total Square Feet",
    fontsize=14
)

plt.ylabel(
    "Price (Lakhs)",
    fontsize=14
)

plt.grid(True)

plt.savefig(
    "static/house_price_graph.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

# =====================================
# GRAPH 2
# PRICE DISTRIBUTION GRAPH
# =====================================

plt.figure(figsize=(12,7))

sns.histplot(
    df['price'],
    bins=30,
    kde=True,
    color='green'
)

plt.title(
    "House Price Distribution",
    fontsize=18,
    fontweight='bold'
)

plt.xlabel(
    "Price (Lakhs)",
    fontsize=14
)

plt.ylabel(
    "Number of Houses",
    fontsize=14
)

plt.grid(True)

plt.savefig(
    "static/price_distribution_graph.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print("Professional graphs updated successfully")