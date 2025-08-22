import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as imb_pipeline


sns.set_theme(rc={"figure.figsize": (7, 4)}, style="darkgrid")


def split_train_data(
    data_X: np.array, data_y: np.array, train_ratio: float = 0.7
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the training data into train and test with the given ratio.

    Args:
        data_df:
            Main train data.
        train_ratio:
            The ratio for train split given between 0 to 1.

    Returns:
        A tuple containing the train and test sets in order X_train, X_test, y_train, y_test.
    """
    train_x, test_x, train_y, test_y = train_test_split(
        data_X,
        data_y,
        train_size=train_ratio,
        random_state=4021,
        shuffle=True,
        # stratify=True,
    )

    return train_x, test_x, train_y, test_y


def train_model(tain_data: pd.DataFrame, model, cv: StratifiedKFold):
    """Train the specified model using the train_data.

    Args:
        train_data:
            The complete training data containing X and y.
        model:
            An instance of the model which needs to be trained.
        cv:
            Cross-Validation instance to be used in training.
    Returns:
        Cross-Val score of the model.
    """
    return


def custom_roc_auc_scorer(estimator, X: np.ndarray, y: np.ndarray) -> float:
    """Custom scoring function to calculate ROC-AUC score."""

    # Probability of the positive class
    y_pred_proba = estimator.predict_proba(X)[:, 1]

    # Return the roc_auc score of the prediction
    return roc_auc_score(y, y_pred_proba)


def evaluate_model_crossval(model, X: np.ndarray, y: np.ndarray):
    """train the model with cross_val_score and return the roc-auc score"""

    # Define the stratified strategy for KFold
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Get the scores
    scores = cross_val_score(model(), X, y, cv=kfold, scoring=custom_roc_auc_scorer)

    # Return the average of scores
    return np.mean(scores)


def evaluate_models(X: np.ndarray, y: np.ndarray, models: list) -> pd.DataFrame:
    """Evaluate the list of procided models with cross_val_scoring.

    Args:
        train_data:
            The training data provided.
        models:
            List of instances of the models which needs to be evaluated.

    Returns:
        A DataFrame with the names of the models as index and cross_val_score as the
        "Score" column.
    """
    model_names = [m.__name__ for m in models]

    scores = [evaluate_model_crossval(m, X, y) for m in models]

    model_scores = pd.DataFrame(data=scores, index=model_names, columns=["Scores"])

    return model_scores


def resample_train_data(X: np.ndarray, y: np.ndarray):
    """Over and under sampling for the training set"""

    # Over sample the minority class
    over_sampler = SMOTE(sampling_strategy=0.1, random_state=42, k_neighbors=5)

    # Under Sample the majority class
    under_sampler = RandomUnderSampler(sampling_strategy=0.7, random_state=42)

    # Make the pipeline
    sampling_steps = [("Over", over_sampler), ("Under", under_sampler)]
    pipline = imb_pipeline(sampling_steps)

    # Fit and resample the data
    X, y = pipline.fit_resample(X, y)

    return X, y


def target_distribution_pie(target_data: np.ndarray, plt_title: str = None):
    """Visualize the distribution of TARGET data."""

    x_bins, counts = np.unique(target_data, return_counts=True)

    g = sns.barplot(x=x_bins, y=counts, width=0.2)
    g.set_title(plt_title)
    plt.show()

    return
