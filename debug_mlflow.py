import mlflow
from mlflow.tracking import MlflowClient

# Load your config
from AdEva_final.config.configuration import Configuration
cfg = Configuration()
ml_cfg = cfg.get_mlflow_config()

mlflow.set_tracking_uri(ml_cfg.tracking_uri)

# Create or retrieve the experiment
exp = mlflow.get_experiment_by_name(ml_cfg.experiment_name)
if exp is None:
    exp_id = mlflow.create_experiment(
        ml_cfg.experiment_name, artifact_location=ml_cfg.tracking_uri
    )
else:
    exp_id = exp.experiment_id

print("Using experiment_id =", exp_id)

with mlflow.start_run(experiment_id=exp_id):
    mlflow.log_param("debug_param", 42)
    mlflow.log_metric("debug_metric", 0.123)
    rid = mlflow.active_run().info.run_id
    print("Logged debug run_id:", rid)
