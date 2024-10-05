import lightgbm as lgb
import numpy as np
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split

from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.lightgbm import TuneReportCheckpointCallback


def train_breast_cancer(config):

    data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)
    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.25)
    train_set = lgb.Dataset(train_x, label=train_y)
    test_set = lgb.Dataset(test_x, label=test_y)
    gbm = lgb.train(
        config,
        train_set,
        valid_sets=[test_set],
        valid_names=["eval"],
        callbacks=[
            TuneReportCheckpointCallback(
                {
                    "binary_logloss": "eval-binary_logloss",
                }
            )
        ],
    )
    preds = gbm.predict(test_x)
    pred_labels = np.rint(preds)
    train.report(
        {
            "mean_accuracy": sklearn.metrics.accuracy_score(test_y, pred_labels),
            "done": True,
        }
    )


if __name__ == "__main__":
    config = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": tune.grid_search(["gbdt", "dart"]),
        "num_leaves": tune.randint(10, 1000),
        "learning_rate": tune.loguniform(1e-8, 1e-1),
    }

    tuner = tune.Tuner(
        train_breast_cancer,
        tune_config=tune.TuneConfig(
            metric="binary_logloss",
            mode="min",
            scheduler=ASHAScheduler(),
            num_samples=2,
        ),
        param_space=config,
    )
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)