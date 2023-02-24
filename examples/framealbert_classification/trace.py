# -*- coding: utf-8 -*-

from cruise import CruiseCLI, CruiseTrainer
from easyguard.appzoo.framealbert_classification.data import FacDataModule
from easyguard.appzoo.framealbert_classification.model import FrameAlbertClassify


if __name__ == "__main__":
    cli = CruiseCLI(FrameAlbertClassify,
                    trainer_class=CruiseTrainer,
                    datamodule_class=FacDataModule,
                    trainer_defaults={},
                    )
    cfg, trainer, model, datamodule = cli.parse_args()
    model.setup("val")
    datamodule.setup("val")

    checkpoint_path = "hdfs://harunava/home/byte_magellan_va/user/wangxian/projects/tts_all_cat_1013/0108_allcat_trans_visionlowlr/model_state_epoch_38000.th"
    export_dir = "./traced_model"
    trainer.trace(model_deploy=model, trace_dataloader=datamodule.val_dataloader(), mode='jit',
                  checkpoint_path=checkpoint_path, export_dir=export_dir)
