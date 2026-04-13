from logging import getLogger
from recbole.utils import init_logger, init_seed, set_color
from recbole.trainer import Trainer
from recbole.config import Config
from recbole.data import data_preparation
from Trainer.lp_single_trainer import SingleTrainer
from Trainer.lp_trainer import LPRecTrainer
from model.LLMCSR_fed import LPRec
from data.lp_dataset import LPFederatedDataset
# turn on if OOM
# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if __name__ == '__main__':
    props = ['./props/OFELLA.yaml']
    dataset_names = ['Arts', 'Office', 'Instruments', 'Games']
    config_dic, dataset_dic, model_dic = {}, {}, {}
    train_data_dic, valid_data_dic, test_data_dic = {}, {}, {}
    local_trainers = {}
    logger = getLogger()
    for dataset_name in dataset_names:
        config = Config(model=LPRec, dataset=dataset_name, config_file_list=props)
        dataset = LPFederatedDataset(config)
        init_seed(config['seed'], config['reproducibility'])
        init_logger(config)
        logger.info(config)
        logger.info(dataset)
        train_data, valid_data, test_data = data_preparation(config, dataset)

        model = LPRec(config, train_data.dataset).to(config['device'])
        logger.info(model)
        trainer = SingleTrainer(config, model)
        config_dic[dataset_name], dataset_dic[dataset_name], model_dic[dataset_name] = config, dataset, model
        train_data_dic[dataset_name], valid_data_dic[dataset_name], test_data_dic[dataset_name] = train_data, valid_data, test_data
        local_trainers[dataset_name] = trainer
    
    fed_trainer = LPRecTrainer(client_trainers = local_trainers, communicatoin_rounds = 5, logger = logger)
    fed_train_metrics = fed_trainer.fedtrain(train_data_dic, valid_data_dic, show_progress=True)
    logger.info(set_color('fed train metrics', 'yellow') + f': {fed_train_metrics}')

    fed_test_metrics = fed_trainer.fedtest(test_data_dic, show_progress=True)
    logger.info(set_color('fed test metrics', 'yellow') + f': {fed_test_metrics}')
