""" import trainer """
from fex.engine.trainer.common_trainer import Trainer
from fex.engine.trainer.xla_trainer import XLATrainer

TRAINER_MAP = {'Trainer': Trainer,
               'XLATrainer': XLATrainer,
               }
