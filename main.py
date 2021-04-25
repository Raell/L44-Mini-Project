from pytorch_lightning.callbacks import ModelCheckpoint, ProgressBar
from tqdm import tqdm

from data import DataGenerator
from models.DSN_model import DSN, MetricsLogger
import pytorch_lightning as pl

from models.expression_model import ExpressionModel


class LitProgressBar(ProgressBar):

    def init_validation_tqdm(self):
        bar = tqdm(
            disable=True,
        )
        return bar


bar = LitProgressBar()


def train_model(model, data, model_file_name):
    logger = MetricsLogger(f"{model_file_name}.json")
    checkpoint_callback = ModelCheckpoint(
        monitor='val_pred_loss',
        filename=model_file_name

    )

    trainer = pl.Trainer(
        gpus=1,
        logger=logger,
        callbacks=[
            bar,
            checkpoint_callback
        ],
        max_epochs=20
    )
    trainer.fit(model, data.train, data.val)
    checkpoint_callback.best_model_path
    trainer.test(model, data.test)
    logger.save_file()

# Examples of model training
# data = DataGenerator()
# train_model(DSN(use_vgg=False, beta=0.05), data, "dsn_untrained")
# train_model(ExpressionModel(), data, "expression_model")


# Loading previously trained models
# data = DataGenerator(AUs=[12])
# model = DSN(use_vgg=False, AUs=[12], beta=0.05)
#
# logger = MetricsLogger(f"test_results.json")
# trainer = pl.Trainer(
#     gpus=1,
#     logger=logger,
#     resume_from_checkpoint="trained_models/AU12_model-v1.ckpt"
# )
#
# trainer.test(model, data.test)
# logger.save_file()