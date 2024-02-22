from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegressionCV, Perceptron
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.neural_network import MLPClassifier
from ucimlrepo import fetch_ucirepo

import pandas as pd
import warnings
import os

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# fetch dataset
students = fetch_ucirepo(id=697)

#
# ASSUMPTIONS & NOTES
# There are 3 classes in the dataset, which are unbalanced.
# There are no missing values in the dataset.
# There are no categorical variables in the dataset, apart from the target variable.
# There are no duplicates in the dataset.
#
X = students.data.features  #
y = students.data.targets  # Classes: Graduate, Enrolled, Dropped

ds = pd.DataFrame(students.data.original)
# Converting to numeric value to calculate correlation
ds["Target"] = ds["Target"].map({"Graduate": 0, "Enrolled": 1, "Dropped": 2})
corr = ds.corr()['Target']
print("CORR\n", ds.corr()["Target"])

# # Keep only the features with correlation above 0.1
# X = X[:, corr[corr > 0.1].index]

# class_freqs = y.value_counts()
# class_weight = class_freqs.max() / class_freqs
# class_weight = {k[0]: v for k, v in class_weight.items()}
# print("CLASS FREQS\n", class_freqs)
# print("CLASS WEIGHTS\n", class_weight)

# class_weight = None       # No class balancing
class_weight = "balanced"  # Auto-balance classes

# metadata
metadata = {key: students.metadata[key] for key in ['name', 'num_instances', 'num_features', 'has_missing_values']}
print("META\n", metadata)

# datasets variable informations
print("VARS\n", students.variables)

import random
RANDOM_STATE = random.randint(0, 1337)
FORMAT_NAME_WIDTH = 27
FORMAT_SCORE_WIDTH = 8

classifiers = {
    # Linear
    "lreg": LogisticRegressionCV(
        random_state=RANDOM_STATE,
        cv=5,
        class_weight=class_weight,
        n_jobs=-1,
    ),
    "svc": SVC(
        kernel="linear",
        C=0.1,
        gamma=0.0001,
        cache_size=2000,  # 1 GiB Cache
        class_weight=class_weight,
        probability=True,
        random_state=RANDOM_STATE,
    ),
    "knn": KNeighborsClassifier(
        n_neighbors=9,
        leaf_size=10,
        n_jobs=-1,
    ),
    "dt": DecisionTreeClassifier(
        max_depth=8,
        max_features=None,
        min_samples_split=2,
        random_state=RANDOM_STATE,
        class_weight=class_weight,
    ),
    "xgb": GradientBoostingClassifier(
        max_depth=5,
        max_features=5,
        min_samples_split=2,
        n_estimators=300,
        random_state=RANDOM_STATE,
    ),
    "rf": RandomForestClassifier(
        bootstrap=False,
        max_depth=20,
        max_features=3,
        min_samples_split=2,
        n_estimators=100,
        random_state=RANDOM_STATE,
        class_weight=class_weight,
        n_jobs=-1,
    ),
    "adabst": AdaBoostClassifier(algorithm="SAMME"),
    "perceptron": Perceptron(
        random_state=RANDOM_STATE,
        penalty="l2",
        class_weight=class_weight,
        n_jobs=-1,
    ),
    "mlp": MLPClassifier(
        hidden_layer_sizes=(
            8,
            8,
            8,
        ),
        random_state=RANDOM_STATE,
    ),
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# rf_param_grid = {
#     "bootstrap": [False, True],
#     "max_depth": [5, 8, 10, 20],
#     "max_features": [3, 4, 5, None],
#     "min_samples_split": [2, 10, 12],
#     "n_estimators": [100, 200, 300],
# }

# dt_param_grid = {
#     "criterion": ["gini", "entropy"],
#     "max_depth": [5, 8, 10, 20],
#     "max_features": [3, 4, 5, None],
#     "min_samples_split": [2, 10, 12],
# }

# xgb_param_grid = {
#     "learning_rate": [0.1, 0.2, 0.3],
#     "n_estimators": [100, 200, 300],
#     "max_depth": [5, 8, 10, 20],
#     "max_features": [3, 4, 5, None],
#     "min_samples_split": [2, 10, 12],
# }

# mlp_param_grid = {
#     "hidden_layer_sizes": [(8, 8), (8, 8, 8), (8, 8, 8, 8)],
#     "activation": ["logistic", "tanh", "relu"],
#     "alpha": [0.0001, 0.001, 0.01],
# }

# perceptron_param_grid = {
#     "penalty": ["l1", "l2", "elasticnet"],
#     "alpha": [0.0001, 0.001, 0.01],
# }

# svc_param_grid = {
#     "C": [0.1, 1, 10],
#     "kernel": ["rbf", "linear"],
#     "gamma": [0.0001, 0.001, 0.01],
# }

# knn_param_grid = {
#     "n_neighbors": [3, 5, 7, 9],
#     "weights": ["uniform", "distance"],
#     "leaf_size": [10, 20, 30, 40],
# }

# # For Hyperparameter tuning
# clf = GridSearchCV(
#     estimator=classifiers["svc"], param_grid=svc_param_grid, cv=5, n_jobs=-1, verbose=1
# )

# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print("Accuracy: ", accuracy_score(y_test, y_pred))
# print(clf.best_params_)
# print(clf.best_estimator_)
# exit(99) # Exit early since im optimizing one model at a time

scores = {}
print(f"{'Classifier Name':<{FORMAT_NAME_WIDTH}} {'Accuracy':>{FORMAT_SCORE_WIDTH}}")
for name, classifier in classifiers.items():
    model = classifier.__class__.__name__
    pipeline = make_pipeline(StandardScaler(), classifier)
    # pipeline.fit(X_train, y_train)
    # score = pipeline.score(X_test, y_test)
    # scores[name] = score

    scores[name] = cross_val_score(pipeline, X, y, cv=5).mean()
    print(f"{model:<{FORMAT_NAME_WIDTH}} {scores[name]:>{FORMAT_SCORE_WIDTH}.02%}")

# Keep only scores above 76%
scores = {k: v for k, v in scores.items() if v > 0.75}
# Keep only classifiers with scores above 76%
classifiers = {k: v for k, v in classifiers.items() if k in scores.keys()}
# Print the best classifiers
print(f"Classifier over thresh (> 0.75): {classifiers.keys()}")

print(f"{'':-<{FORMAT_NAME_WIDTH + FORMAT_SCORE_WIDTH + 1}}")

voting_classifier = VotingClassifier(
    estimators=[(name, classifier) for name, classifier in classifiers.items()],
    voting="hard",
    n_jobs=-1,
)
pipeline = make_pipeline(StandardScaler(), voting_classifier)
score = cross_val_score(pipeline, X, y, cv=5).mean()
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

print(f"{'VotingClassifier':<{FORMAT_NAME_WIDTH}} {score:>{FORMAT_SCORE_WIDTH}.02%}")
print(f"{'':-<{FORMAT_NAME_WIDTH + FORMAT_SCORE_WIDTH + 1}}")
report = classification_report(y_test, predictions)
print("Classification Report:\n", report)
