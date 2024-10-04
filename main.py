## Import Modules
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping

## Load the Data
data = pd.read_csv("train.csv")

NUM_EXAMPLES = 3116945

X = data.drop(columns=["class"]).set_index("id")[:NUM_EXAMPLES]
y = (data["class"] == "p").astype(int)[:NUM_EXAMPLES]

X_test = pd.read_csv("test.csv").set_index("id")

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.05)

## Build the Preprocessor

# dropped_cols = ["spore-print-color", "veil-color", "veil-type", 
# 	"stem-surface", "stem-root"]

dropped_cols = []

categorical_cols = [col for col in X.columns if X[col].dtype == "object" and 
	col not in dropped_cols]

numerical_cols = [col for col in X.columns if X[col].dtype in ["int64", "float64"] and 
	col not in dropped_cols]

categorical_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="constant")),
	("encoder", OneHotEncoder(handle_unknown="ignore"))])

numerical_pipeline = Pipeline(steps=[("imputer", SimpleImputer()), ("scaler", StandardScaler())])

preprocessor = ColumnTransformer([("categorical", categorical_pipeline, categorical_cols),
	("numerical", numerical_pipeline, numerical_cols)])

## Build the Model

callback = EarlyStopping(rounds=10, save_best=True)
model = XGBClassifier(n_estimators=1000, max_depth=15, callbacks=[callback], 
	eval_metric="error", n_jobs=8)

## Build and Fit the Final Pipeline
pipeline = Pipeline(steps=[("preprocessor", preprocessor),
	("model", model)])

# print("Starting to train...")

preprocessor.fit(X_train, y_train)
pipeline.fit(X_train, y_train, model__eval_set=[(preprocessor.transform(X_valid), y_valid)],
	model__verbose=1)

## Making Final Predictions

def out(y_hat):
	if y_hat == 1:
		return 'p'
	return 'e'

# print("Starting to predict...")

predictions = list(map(out, pipeline.predict(X_test)))
result = pd.DataFrame({"class": predictions}, index=X_test.index)

result.to_csv("submission_1.csv")




