from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

from mlflow.tracking import MlflowClient
from mlflow.artifacts import download_artifacts

import mlflow
import mlflow.sklearn
import shap
import numpy as np
import os


if __name__ == "__main__":
    # prepare training data
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X = X.iloc[:50, :8]
    y = y.iloc[:50]
    # log an explanation
    with mlflow.start_run() as run:
    
        # train a model
        model = RandomForestClassifier()
        model.fit(X, y)
        score = model.score(X, y)
        print("Score: %s" % score)
        mlflow.log_metric("score", score)
        mlflow.sklearn.log_model(model, "model")
        print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
        mlflow.shap.log_explanation(lambda X: model.predict_proba(X)[:, 1], X)

# list artifacts
client = MlflowClient()
artifact_path = "model_explanations_shap"
artifacts = [x.path for x in client.list_artifacts(run.info.run_id, artifact_path)]
print("# artifacts:")
print(artifacts)

# load back the logged explanation
dst_path = download_artifacts(run_id=run.info.run_id, artifact_path=artifact_path)
base_values = np.load(os.path.join(dst_path, "base_values.npy"))
shap_values = np.load(os.path.join(dst_path, "shap_values.npy"))

# show a force plot
shap.force_plot(float(base_values), shap_values[0, :], X.iloc[0, :], matplotlib=True)
