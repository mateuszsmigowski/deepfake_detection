from src.loaders.config import ConfigModel, ExperimentConfig
from src.pipelines.data.split import OneOutSplitModel
from src.models.dann.model import DANNClassifier
from src.training import configure_reproducibility, DANNTrainer, DANNAdaptationTrainer
from src.pipelines.helpers import prepare_records, prepare_data_loader

class DANNPipeline:

    def __init__(self, config: ConfigModel, one_out_split: OneOutSplitModel):
        self.config = config
        self.one_out_split = one_out_split

    def run(self):

        configure_reproducibility(self.config.runtime.seed, self.config.runtime.deterministic)
        train_records, target_train, val_records, test_records = prepare_records(self.config, self.one_out_split)

        source_domain_to_int = self.config.experiment.source_domain_to_int
        target_domain_to_int = self.config.experiment.target_domain_to_int

        source_train_loader = prepare_data_loader(
            self.config,
            train_records,
            shuffle=True,
            domain_to_int=source_domain_to_int,
            isTraining=True,
        )
        target_train_loader = prepare_data_loader(
            self.config,
            target_train,
            shuffle=True,
            domain_to_int=target_domain_to_int,
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
            domains_count=self._get_domains_count(),
        )

        match self.config.experiment.protocol:
            case ExperimentConfig.Protocol.GENERALIZATION:
                trainer = DANNTrainer(
                    self.config,
                    classifier,
                    source_train_loader,
                    val_loader,
                    test_loader,
                )
            case ExperimentConfig.Protocol.ADAPTATION:
                trainer = DANNAdaptationTrainer(
                    self.config,
                    classifier,
                    source_train_loader,
                    target_train_loader,
                    val_loader,
                    test_loader,
                )

        trainer.run()

    def _get_domains_count(self) -> int:

        match self.config.experiment.protocol:
            case ExperimentConfig.Protocol.GENERALIZATION:
                return len(self.config.experiment.source_domain_to_int)
            case ExperimentConfig.Protocol.ADAPTATION:
                return 2 # TODO: Remove magic number, should be calculated