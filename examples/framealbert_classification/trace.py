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
    model.setup(stage="val")

    datamodule.setup(stage="val")
    trace_loader = datamodule.val_dataloader()

    checkpoint_path = ''
    export_dir = "./traced_model"

    trainer.trace(
        model_deploy=model,
        trace_dataloader=trace_loader,
        mode='anyon',
        checkpoint_path=checkpoint_path,
        export_dir=export_dir
    )
