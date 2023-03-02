import os
import sys
import os.path as osp

sys.path.append("/opt/tiger/cruise")

from addict import Dict
from cruise.data_module.cruise_loader import DistributedCruiseDataLoader
from cruise.data_module.preprocess.create_preprocess import parse_cruise_processor_cfg
from cruise.data_module.preprocess.decode import TFApiExampleDecode

from examples.live_gandalf.utils.driver import get_logger, TRIAL_ID
from examples.live_gandalf.utils.util import scan_hdfs_dir, scan_local_dir, safe_int
import math


data_type_map = {
    "ParquetDataFactory": "parquet",
    "TFRecordDataFactory": "tfrecord",
    "JsonLDataFactory": "jsonl",
    "KVDataFactory": "kv",
}


def get_ds_path(
        folder_path_str,
        folder_str,
        data_type,
        fname_pattern,
        fmin_size=None,
        group_keys=None,
        shuffle=False,
):
    if not fmin_size:
        fmin_size = None
    data_type = data_type_map[data_type]
    if not fname_pattern:
        if data_type == "jsonl":
            fname_pattern = ""
        elif data_type == "kv":
            fname_pattern = "*index"
        else:
            fname_pattern = "*{}".format(data_type)
    folders = folder_str.split("$")
    if "$" in folder_path_str:
        folder_paths = folder_path_str.split("$")
    else:
        folder_paths = [folder_path_str] * len(folders)
    regex_type = "shell"
    data_sources, data_types = [], []
    for folder_path, folder in zip(folder_paths, folders):
        if data_type == "tfrecord":
            # for tfrecord, follow the logic in tfrecord data factory
            full_path = osp.join(folder_path, folder, fname_pattern)
            folder_path, folder, fname_pattern = full_path.rsplit("/", 2)
        if folder_path.startswith("hdfs"):
            cur_data_source = scan_hdfs_dir(
                folder_path,
                folder,
                fname_pattern,
                stick_folder_file=data_type == "kv",
                min_size=fmin_size,
            )
        else:
            cur_data_source = []
            for cur_folder in folder.split("|"):
                cur_data_source += scan_local_dir(folder_path, cur_folder, fname_pattern)
        # remove duplicate files
        cur_data_type = data_type
        cur_data_source = list(set(cur_data_source))
        cur_data_source.sort()
        if shuffle:
            import random
            random.Random(safe_int(TRIAL_ID or 42)).shuffle(cur_data_source)
        if data_type == "kv":
            cur_data_source = [x[:-6] for x in cur_data_source]
            if group_keys:
                for key in group_keys:
                    _tmp = [x for x in cur_data_source if key in x]
                    data_sources.append(_tmp)
                    data_types.append(cur_data_type)
        else:
            data_sources.append(cur_data_source)
            data_types.append(cur_data_type)
    return data_sources, data_types


def create_cruise_process_config(cfg, mode, is_kv=False):
    process_cfg_dict = {
        "custom_modals": "maxwell_modal",
        "custom_op_modules": {
            "maxwell_modal": "dataset.feature_provider",
        },
        "modal_keys": {
            "maxwell_modal": "all",
        },
        "custom_transforms": {
            "maxwell_modal": {
                "transform": [],
                "batch_transform": [],
                "skip_collate": True,
            }
        },
    }
    if is_kv:
        process_cfg_dict["custom_transforms"]["maxwell_modal"]["transform"].append({"CruiseKVFeatureProvider": Dict(cfg.feature_provider)})
        process_cfg_dict["custom_transforms"]["maxwell_modal"]["batch_transform"].append({"CruiseKVBatchFeatureProvider": Dict(cfg.feature_provider)})
        if mode == "train":
            process_cfg_dict["custom_transforms"]["maxwell_modal"]["transform"][0]["CruiseKVFeatureProvider"]["save_extra"] = False
            process_cfg_dict["custom_transforms"]["maxwell_modal"]["batch_transform"][0]["CruiseKVBatchFeatureProvider"]["save_extra"] = False
        else:
            process_cfg_dict["custom_transforms"]["maxwell_modal"]["transform"][0]["CruiseKVFeatureProvider"]["save_extra"] = True
            process_cfg_dict["custom_transforms"]["maxwell_modal"]["transform"][0]["CruiseKVFeatureProvider"]["eval_mode"] = True
            process_cfg_dict["custom_transforms"]["maxwell_modal"]["batch_transform"][0]["CruiseKVBatchFeatureProvider"]["save_extra"] = True
            process_cfg_dict["custom_transforms"]["maxwell_modal"]["batch_transform"][0]["CruiseKVBatchFeatureProvider"]["eval_mode"] = True

    else:
        process_cfg_dict["custom_transforms"]["maxwell_modal"]["transform"].append({"CruiseFakeProcessor": {}})
        process_cfg_dict["custom_transforms"]["maxwell_modal"]["batch_transform"].append({"CruiseFeatureProvider": Dict(cfg.feature_provider)})
        if mode == "train":
            process_cfg_dict["custom_transforms"]["maxwell_modal"]["batch_transform"][0]["CruiseFeatureProvider"]["save_extra"] = False
        else:
            process_cfg_dict["custom_transforms"]["maxwell_modal"]["batch_transform"][0]["CruiseFeatureProvider"]["save_extra"] = True
            process_cfg_dict["custom_transforms"]["maxwell_modal"]["batch_transform"][0]["CruiseFeatureProvider"]["eval_mode"] = True

    return Dict(process_cfg_dict)


def create_cruise_dataloader(
        cfg, df_type, data_input_dir, data_folder, arg_dict, mode="val", specific_bz=None
):
    get_logger().info("CruiseLoader is on...")
    arg_dict_cp = Dict(arg_dict)
    assert mode in [
        "train",
        "val",
        "test",
        "trace",
    ], f"mode should be one of [train, val, test], instead of {mode}"
    data_sources, data_types = get_ds_path(
        data_input_dir,
        data_folder,
        df_type,
        arg_dict_cp.filename_pattern,
        arg_dict_cp.file_min_size,
        arg_dict_cp.group_keys,
        arg_dict_cp.shuffle_files
    )
    ds_num = len(data_sources)
    drop_last = arg_dict_cp.get("drop_last", True)
    shuffle = arg_dict_cp.get("shuffle", True)
    fast_resume = arg_dict_cp.get("fast_resume", True)
    parquet_cache_on = arg_dict_cp.get("parquet_cache_on", True)
    batch_size = arg_dict_cp.batch_size
    predefined_steps = -1
    use_arnold = True

    if mode == "train":
        predefined_steps = cfg.trainer.train_max_iteration

    if mode == "val":
        drop_last = False
        # trick only in val: half bz to lower mem usage
        if arg_dict_cp.batch_size_val == -1:
            get_logger().info(
                "batch_size_val is not set, use batch_size // 2 as default"
            )
            arg_dict_cp.batch_size_val = arg_dict_cp.batch_size // 2
        batch_size = arg_dict_cp.batch_size_val
        predefined_steps = cfg.trainer.test_max_iteration

    if mode == "test":
        drop_last = False
        shuffle = False
        predefined_steps = cfg.tester.max_iteration

    if mode == "trace":
        if "ParquetDataFactory" in df_type:
            shuffle = False

    # use in trace model
    if specific_bz and isinstance(specific_bz, int):
        batch_size = specific_bz

    num_workers = arg_dict_cp.num_workers
    num_readers = [arg_dict_cp.num_parallel_reads] * ds_num
    multiplex_dataset_weights = arg_dict_cp.multiplex_dataset_weights
    # whether to mix a batch with data from different dataset, default to be False/None
    multiplex_mix_batch = arg_dict_cp.multiplex_mix_batch
    is_kv = data_types[0] == "kv"
    if is_kv:
        multiplex_mix_batch = True
        use_arnold = num_workers > 0

    if multiplex_mix_batch:
        # for the case when one batch data is mixed by multiple datasets
        if not multiplex_dataset_weights:
            batch_sizes = [batch_size // ds_num] * ds_num
            remain = batch_size - sum(batch_sizes)
            for i in range(remain):
                batch_sizes[i] += 1
        else:
            batch_sizes = [
                math.floor(batch_size * p) for p in multiplex_dataset_weights
            ]
            remain = batch_size - sum(batch_sizes)
            for i in range(remain):
                batch_sizes[i] += 1
        multiplex_dataset_weights = []
    else:
        # for the case when one batch data is from only single dataset each time,
        # while the dataset is chosen randomly from all the given datasets
        if not multiplex_dataset_weights:
            if ds_num > 1:
                # read each dataset with equal probability when multiplex_dataset_weights is not given
                multiplex_dataset_weights = [1 / ds_num] * ds_num
            else:
                # since we only have one dataset, the multiplex_dataset_weights does not affcet the loading logic
                # we make it to be an empty list here to match the original logic for single dataset
                multiplex_dataset_weights = []
        batch_sizes = [batch_size] * ds_num
    process_cfg = create_cruise_process_config(cfg, mode, is_kv)
    cruise_processor = parse_cruise_processor_cfg(process_cfg, "")
    keys_or_columns = []
    last_step = 0
    if ds_num > 1 and predefined_steps == -1:
        predefined_steps = "max"

    if data_types[0] == "tfrecord":
        features = arg_dict_cp.data_schema
        enable_tf_sample_sharding = int(os.getenv("CRUISE_ENABLE_TF_SAMPLE_SHARDING", "0"))
        to_numpy = not enable_tf_sample_sharding
        decode_fn = [TFApiExampleDecode(features=features, key_mapping=dict(), to_numpy=to_numpy)] * ds_num
    elif is_kv and not use_arnold:
        # since kv feature provider would get the index 0 of the given data; while in cruise loader, if reading kv data from 
        # torch loader, the output data would be the data itself, not a list. Here we make it a list by using decode fn, to 
        # ensure it is runnable, but this might be a little bit hacky.
        decode_fn = [lambda x: [x]] * ds_num
    else:
        decode_fn = None

    loader = DistributedCruiseDataLoader(
        data_sources,
        keys_or_columns,
        batch_sizes,
        num_workers,
        num_readers,
        decode_fn,
        cruise_processor,
        predefined_steps,
        data_types,
        last_step,
        shuffle=shuffle,
        multiplex_weights=multiplex_dataset_weights,
        drop_last=drop_last,
        use_arnold=use_arnold,
        transform_replace_all=is_kv,
        fast_resume=fast_resume,
        parquet_cache_on=parquet_cache_on,
    )
    return loader
