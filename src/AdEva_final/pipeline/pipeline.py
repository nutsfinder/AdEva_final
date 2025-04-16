from AdEva_final.config.configuration import Configuration
from AdEva_final.pipeline.data_ingestion import DataIngestion
from AdEva_final.pipeline.data_transformation import DataTransformation
from AdEva_final.pipeline.model_trainer import ModelTrainer
from AdEva_final.pipeline.evaluation import Evaluation

def run_pipeline():
    # 1. Load configs
    cfg = Configuration()
    di_cfg = cfg.get_data_ingestion_config()
    dt_cfg = cfg.get_data_transformation_config()
    mt_cfg = cfg.get_model_training_config()
    ev_cfg = cfg.get_evaluation_config()

    # 2. Ingest
    ingestor = DataIngestion(di_cfg)
    train_path, test_path = ingestor.ingest()

    # 3. Transform
    transformer = DataTransformation(dt_cfg, cfg.params)
    X_train, y_train, X_test, y_test = transformer.transform(train_path, test_path)

    # 4. Train
    trainer = ModelTrainer(mt_cfg, cfg.params)
    model = trainer.train(X_train, y_train)

    # 5. Evaluate
    evaluator = Evaluation(ev_cfg)
    evaluator.evaluate(model, X_test, y_test)

if __name__ == "__main__":
    run_pipeline()
