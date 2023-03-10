# -*- coding:utf-8 -*-
# Email:    jiangxubin@bytedance.com
# Created:  2023-03-01 12:52:19
# Modified: 2023-03-01 12:52:19
import os
import re
import warnings
import argparse
import logging
from addict import Dict
from cruise import CruiseTrainer, CruiseCLI
from cruise.utilities.rank_zero import rank_zero_info
from cruise.trainer.logger import TensorBoardLogger
from cruise.trainer.logger.tracking import TrackingLogger
from easyguard.utils.arguments import print_cfg
from examples.live_gandalf.builder import get_model,get_feature_provider
from utils.config import config
from utils.driver import reset_logger, get_logger, init_env,init_device, DIST_CONTEXT
from utils.util import load_conf, load_from_tcc,load_from_bbc,check_hdfs_exist,hmkdir,check_config,update_config,init_seeds


class MainPipeline:
    def __init__(self, config, local_rank=-1, debug=False, seed=42):
        self.config = config
        self._workflow_conf = self.config.workflow
        self._trainer_conf = self.config.trainer
        self._test_conf = self.config.tester
        self._trace_conf = self.config.tracer
        self._trainer_defaults = None
        self.config_trainer(local_output_dir,hdfs_output_dir,train_data_loader,test_data_loader,train_kwargs)

        # set device
        self._local_rank = local_rank
        init_device(local_rank=self._local_rank)

        # set seeds (must be set after init_device due to global_rank initialization from DIST_CONTEXT)
        init_seeds(seed + DIST_CONTEXT.global_rank, cuda_deterministic=True)

        # parse folder info
        self._input_dir = self._workflow_conf.input_dir.replace("hdfs:///user", "hdfs://haruna/user")
        self._val_input_dir = self._workflow_conf.get("val_input_dir", "").replace("hdfs:///user", "hdfs://haruna/user")
        self._test_input_dir = self._workflow_conf.get("test_input_dir", "").replace("hdfs:///user", "hdfs://haruna/user")
        self._local_output_dir = self._workflow_conf.local_output_dir
        self._hdfs_output_dir = self._workflow_conf.hdfs_output_dir
        self._train_folder = self._workflow_conf.train_folder
        self._val_folder = self._workflow_conf.val_folder
        self._test_folder = self._workflow_conf.test_folder
        self._trace_folder = self._workflow_conf.trace_folder
        self._additional_input_dir = self._workflow_conf.get("additional_input_dir", "").replace("hdfs:///user", "hdfs://haruna/user")
        self._additional_input_folder = self._workflow_conf.additional_input_folder

        # setup local and remote folders
        os.makedirs(self._local_output_dir, exist_ok=True)
        if not check_hdfs_exist(self._hdfs_output_dir):
            hmkdir(self._hdfs_output_dir)
        if not check_hdfs_exist(self._hdfs_output_dir + '/bin'):
            hmkdir(self._hdfs_output_dir + '/bin')

            # init elements
        self._init_dataset()
        self._init_model()
        # set logger
        self._set_logger(debug=debug)
        get_logger().info("Config:\n {}".format(self.config))
        # helps Arnold track usage analytics
        try:
            import pickle
            if DIST_CONTEXT.local_rank == 0:
                with open('global.config', 'wb') as f:
                    pickle.dump(self.config, f)
        except:
            pass

    def _set_logger(self, debug=False):
        logger = get_logger()
        if self._local_rank in [-1, 0]:
            if debug:
                log_level = logging.DEBUG
            else:
                log_level = logging.INFO
        else:
            log_level = logging.ERROR
        logger.setLevel(log_level)
        reset_logger(logger)

    def _init_model(self):
        model_instance_conf = Dict(self.config.model_instance)
        model_type = model_instance_conf.pop("type")
        self.model = get_model(model_type)

    def _init_dataset(self):
        feature_provider_conf = Dict(self.config.feature_provider)
        fp_type = feature_provider_conf.pop("type")
        self.train_dataset = get_feature_provider(fp_type)
        self.test_dataset = get_feature_provider(fp_type)

    def init_loggers(self,train_kwargs):
        log_path = './events_log'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cruise_loggers = ['console', TensorBoardLogger(save_dir=log_path, hdfs_dir=None, name='',
                flush_logs_every_n_steps=max(1, 100 // self.summary_interval),
                version='', ignore_keys=["__elapsed__"])]
        try:
            # add tracking logger
            if train_kwargs.get('tracking_project_name', ''):
                tracking_project_name = train_kwargs['tracking_project_name']
                if "/" in tracking_project_name:
                    project, name = tracking_project_name.rsplit("/", 1)
                    project = project.replace('/', '_')
                    # remove special chars
                    name = "_".join(re.findall(r'[a-zA-Z0-9\u4E00-\u9FA5-_./@]{1,128}',  name))
                else:
                    project = tracking_project_name
                    name = ""
                cruise_loggers.append(
                    TrackingLogger(
                        project=project,
                        name=name,
                        config={'trainer': train_kwargs},
                        version='', ignore_keys=["__elapsed__"],
                        allow_keys=['training/grad_norm']))
                rank_zero_info("Tracking enabled with name: {}".format(tracking_project_name))
        except ImportError:
            rank_zero_info("Tracking not enabled")
        return cruise_loggers

    def init_callbacks(self,train_kwargs):
        pass

    def config_trainer(self,hdfs_output_dir,train_kwargs):
        self.auto_resume = train_kwargs.auto_resume
        hdfs_ckpt_path = os.path.join(hdfs_output_dir, 'checkpoints')
        max_steps = train_kwargs.get("max_total_iter", -1)
        if max_steps == -1:
            # default max_total_iter equals to epochs * train_max_iteration if not set
            if train_kwargs.get("epochs", 1) > 0 and train_kwargs.get("train_max_iteration", -1) > 0:
                max_steps = train_kwargs.get("epochs", 1) * train_kwargs.get("train_max_iteration", -1)
            else:
                # unable to set the max_total_iter, set to unlimited
                max_steps = -1
        cruise_loggers = self.init_loggers(train_kwargs)
        # callback_list = self.init_callbacks(train_kwargs)
        self._trainer_defaults = {
                "logger": cruise_loggers,
                "log_every_n_steps": 50,
                "enable_versions":True,
                "precision": 16 if train_kwargs.enable_amp else 32,
                "max_epochs": train_kwargs.get("epochs", 1),
                "max_steps": max_steps,
                "limit_train_batches":train_kwargs.get("train_max_iteration", -1),
                "limit_val_batches":train_kwargs.get("test_max_iteration", -1),
                "val_check_interval":int(train_kwargs.get('output_iteration', 50)),
                "gradient_clip_val":train_kwargs.clip_grad_norm,
                "summarize_model_depth": 3,
                "resume_ckpt_path":hdfs_ckpt_path if self.auto_resume else None,
                "resume_loader_state":train_kwargs.get('resume_dataloader', False) or self.auto_resume,
                "project_name": 'augustus',
                "experiment_name": "None",
                }

    def run(self, enable_train, enable_test, trace_model):
        assert (enable_train or enable_test or trace_model), "train or test or trace? please choose at least one of them!"
        if enable_train:
            get_logger().info("Train Mode starts...")
            cli = CruiseCLI(self.model,
                            trainer_class=CruiseTrainer,
                            datamodule_class=self.train_dataset,
                            trainer_defaults=self._trainer_defaults)
            cfg, trainer, model, datamodule = cli.parse_args()
            print_cfg(cfg)
            trainer.fit(model, datamodule)
        if enable_test:
            get_logger().info("Test Mode starts...")
            cli = CruiseCLI(self.model,
                            trainer_class=CruiseTrainer,
                            datamodule_class=self.test_dataset,
                            trainer_defaults=self._test_conf)
            cfg, trainer, model, datamodule = cli.parse_args()
            print_cfg(cfg)
            trainer.predict(model, datamodule)
        if trace_model:
            get_logger().info("Trace Model starts...")
            raise KeyError('"trace_model" has not been implemented')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, help="local conf")
    parser.add_argument("--config", type=str, help="local yaml conf")
    parser.add_argument("--tcc_key", type=str, help="TCC conf")
    parser.add_argument("--bbc_key", type=str, help="BBC conf")
    parser.add_argument("--enable_train", action="store_true")
    parser.add_argument("--enable_test", action="store_true")
    parser.add_argument("--trace_model", action="store_true")
    parser.add_argument("--debug", action="store_true", help="show DEBUG level log")
    parser.add_argument("--hdfs_jdk_heap_max_size",type=str,default="2g",help="JDK heap size application for hdfs, default 768m, could also be 1g/2g/3g..")
    parser.add_argument("--local_rank", type=int, default=-1, help="DDP mode gives by default")
    parser.add_argument("--seed", type=int, default=42, help="train seed")
    # parse
    args, config_override = parser.parse_known_args()
    enable_train = args.enable_train
    enable_test = args.enable_test
    trace_model = args.trace_model
    debug = args.debug
    hdfs_jdk_heap_max_size = args.hdfs_jdk_heap_max_size
    # setup env
    init_env(hdfs_jdk_heap_max_size=hdfs_jdk_heap_max_size)
    if args.conf:
        conf = Dict(load_conf(args.conf))
        project_name = args.conf
    elif args.tcc_key:
        conf = Dict(load_from_tcc(args.tcc_key,
                                  tcc_psm=os.environ.get("MAXWELL_TCC_PSM",
                                                         "data.account.content_security_multimodal")))
        project_name = args.tcc_key
    elif args.bbc_key:
        conf = Dict(load_from_bbc(args.bbc_key,
                                  bbc_psm=os.environ.get("MAXWELL_BBC_PSM", "data.content.maxwell")))
        project_name = args.bbc_key
    else:
        raise KeyError("should provide a local conf or tcc_key or bbc_key")
    # preprocess config
    config.trainer.tracking_project_name = project_name  # default monitor settings
    config.update(conf)
    update_config(config, config_override)
    check_config(config, enable_train, enable_test, trace_model)
    # trigger workflow
    main_workflow = MainPipeline(config, local_rank=int(os.getenv('LOCAL_RANK', -1)), debug=debug, seed=args.seed)
    main_workflow.run(
        enable_train=enable_train,
        enable_test=enable_test,
        trace_model=trace_model,
    )

