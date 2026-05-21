from src.loaders.config import ConfigModel
from src.pipelines.data.split import OneOutSplitModel
from src.models.dann.model import DANNClassifier
from src.training import configure_reproducibility, DANNTrainer
from src.pipelines.helpers import prepare_records, prepare_data_loader

class DANNPipeline:

    def __init__(self, config: ConfigModel, one_out_split: OneOutSplitModel):
        self.config = config
        self.one_out_split = one_out_split

    def run(self):

        configure_reproducibility(self.config.runtime.seed, self.config.runtime.deterministic)
        train_records, val_records, test_records = prepare_records(self.config.data, self.one_out_split)

        source_domain_to_int = self.config.experiment.source_domain_to_int

        train_loader = prepare_data_loader(
            self.config,
            train_records,
            shuffle=True,
            domain_to_int=source_domain_to_int,
            isTraining=True,
        )
        val_loader = prepare_data_loader(
            self.config,
            val_records,
            shuffle=False,
            domain_to_int=source_domain_to_int,
        )
        test_loader = prepare_data_loader(
            self.config,
            test_records,
            shuffle=False,
        )
        classifier = DANNClassifier(
            self.config.model,
            self.config.domain_adaptation,
            domains_count=len(source_domain_to_int),
        )
        trainer = DANNTrainer(self.config, classifier, train_loader, val_loader, test_loader)
        trainer.run()