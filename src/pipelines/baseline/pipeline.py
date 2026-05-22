from src.pipelines.data.split import OneOutSplitModel
from src.loaders.config import ConfigModel
from src.models.baseline.model import BaselineClassifier
from src.training import configure_reproducibility, BaselineTrainer
from src.pipelines.helpers import prepare_records, prepare_data_loader


class BaselinePipeline:

    def __init__(self, config: ConfigModel, one_out_split: OneOutSplitModel):

        self.config = config
        self.one_out_split = one_out_split

    def run(self):

        configure_reproducibility(self.config.runtime.seed, self.config.runtime.deterministic)

        train_records, _, val_records, test_records = prepare_records(self.config, self.one_out_split)

        train_loader = prepare_data_loader(self.config, train_records, shuffle=True, isTraining=True)
        val_loader = prepare_data_loader(self.config, val_records, shuffle=False)
        test_loader = prepare_data_loader(self.config, test_records, shuffle=False)
        classifier = BaselineClassifier(self.config.model)
        trainer = BaselineTrainer(self.config, classifier, train_loader, val_loader, test_loader)
        trainer.run()