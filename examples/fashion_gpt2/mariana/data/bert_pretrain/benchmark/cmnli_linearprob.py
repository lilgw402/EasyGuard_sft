"""CMNLI classification, the pretrained model encoder itself is freezed.
The filename is linear prob but actual model is a two layer MLP with dropout and tanh activation."""
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import numpy as np
import subprocess
import logging

import cruise
from cruise import CruiseModule, CruiseTrainer
from cruise.data_module import DistributedCruiseDataLoader
from cruise.utilities import DIST_ENV
from cruise.utilities.hdfs_io import hdfs_open, hglob, hopen, hmkdir
from cruise.utilities.cloud_io import load as torch_io_load
from cruise.utilities import move_data_to_device
from mariana.optim.optimizer import AdamW  # diff to pytorch native AdamW?
from mariana.optim.lr_scheduler import get_linear_schedule_with_warmup  # use compose?


def save(obj, filepath: str, **kwargs):
    """ save model """
    if filepath.startswith("hdfs://"):
        with hopen(filepath, "wb") as writer:
            torch.save(obj, writer, **kwargs)
    else:
        torch.save(obj, filepath, **kwargs)


class NLIProcessor:
    def __init__(self, batch_size, padding_index: int = 2):
        self._pad_id = padding_index
        # bsz is required to ensure zero3 mode all batches have same shape
        self._bsz = batch_size

    def transform(self, data_dict):
        input_ab = data_dict['input_ids']
        input_ab_mask = data_dict['input_mask']
        label_id = data_dict['label']
        input_ab_segment = data_dict['segment_ids']
        pos_ids = list(range(len(input_ab)))

        instance_data = {
            'input_ids': torch.tensor(input_ab).long(),
            'input_mask': torch.tensor(input_ab_mask).long(),
            'segment_ids': torch.tensor(input_ab_segment).long(),
            'position_ids': torch.tensor(pos_ids).long(),
            'labels': torch.tensor(int(label_id)).long(),
        }
        return instance_data

    def batch_transform(self, batch_data):
        # stack tensors in dicts
        result_data = {}
        if len(batch_data) < self._bsz:
            offset = self._bsz - len(batch_data)
            result_data['_num_valid_samples'] = len(batch_data)
            batch_data += [batch_data[0]] * offset
        else:
            result_data['_num_valid_samples'] = self._bsz

        for key in batch_data[0].keys():
            result_data[key] = torch.stack([dd[key] for dd in batch_data], dim=0)
        return result_data


class PoolingClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, num_labels: int, hidden_size: int = 768, hidden_dropout_prob: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class NLIModel(CruiseModule):
    def __init__(self,
                 num_labels: int = 3,
                 hidden_size: int = 768,
                 lr: float = 1e-3,
                 wd: float = 1e-4,
                 warmup_step_rate: float = 0.1,
                 ):
        super().__init__()
        self.save_hparams()
        # TODO(Zhi): allow NLI model to be customized
        self.dropout = torch.nn.Dropout(0.1)
        self.num_labels = num_labels
        self.classifier = PoolingClassificationHead(num_labels=num_labels, hidden_size=hidden_size, hidden_dropout_prob=0.1)
        self.apply(self._init_weights)

        if self.num_labels == 1:
            self.loss = torch.nn.MSELoss()
        else:
            self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, sequence_output):
        enc_out = self.dropout(sequence_output)
        logits = self.classifier(enc_out)
        return logits

    def training_step(self, batch, batch_idx):
        sequence_output = batch[0]
        label = batch[1].long()
        logits = self.forward(sequence_output)
        if self.num_labels == 1:
            probs = F.sigmoid(logits.squeeze(-1))
        else:
            probs = F.softmax(logits, dim=-1)

        output_dict = {}
        if self.num_labels == 1:
            loss = self.loss(probs, label)
        else:
            loss = self.loss(logits, label.long().view(-1))
            acc = ((label == torch.argmax(probs, dim=-1).long()).sum().float() / label.numel()).item()
            output_dict['cmnli_acc'] = acc
        output_dict['loss'] = loss
        return output_dict

    def validation_step(self, batch, batch_idx):
        sequence_output = batch[0]
        label = batch[1].long()
        mask = label > -1
        preds = torch.argmax(self.forward(sequence_output), dim=-1).long()
        acc = ((label[mask] == preds[mask]).sum().float() / label[mask].numel()).item()
        return {'cmnli_linearprob': acc}

    def configure_optimizers(self):
        no_decay = ['bias', 'bn', 'norm', 'ln', 'attn_', 'Norm']
        no_dacay_params_dict = {'params': [], 'weight_decay': 0.0}
        normal_params_dict = {'params': [], 'weight_decay': self.hparams.wd}

        for n, p in self.named_parameters():
            if any(nd in n for nd in no_decay):
                no_dacay_params_dict['params'].append(p)
            else:
                normal_params_dict['params'].append(p)
        optimizer_grouped_parameters = [
            no_dacay_params_dict,
            normal_params_dict]

        optm = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.lr,
            betas=(0.9, 0.999),
            eps=1e-6,
            weight_decay=self.hparams.wd,
            correct_bias=False  # TODO(Zhi): align with master branch ADAMW
            )

        warmup_steps = self.hparams.warmup_step_rate * self.trainer.total_steps
        lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=optm,
                num_warmup_steps=warmup_steps,
                num_training_steps=self.trainer.total_steps)

        return [optm], [lr_scheduler]

    def lr_scheduler_step(
        self,
        schedulers,
        **kwargs,
    ) -> None:
        r"""
        默认是per epoch的lr schedule, 改成per step的
        """
        # if self.trainer.global_step == 0:
        #     # skip first step
        #     return
        for scheduler in schedulers:
            scheduler.step()

    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class CMNLILinearProbBenchmark:
    def __init__(self, model, output_path,
                 train_path: str = 'hdfs://haruna/home/byte_data_aml_research/user/zhangzhi.joshua/data/cmnli/train_oldcut.parquet',
                 val_path: str = 'hdfs://haruna/home/byte_data_aml_research/user/zhangzhi.joshua/data/cmnli/dev_oldcut.parquet',
                 num_labels=3,
                 hidden_size=None,
                 train_batch_size=64,
                 val_batch_size=64,
                 global_fit_batch_size=64,
                 num_workers=4,
                 trainer_kwargs=None,
                 model_kwargs=None,
                 verbose=False,
                 *args, **kwargs):
        self.backbone_model = model
        if hidden_size is not None:
            encoder_dim = hidden_size
        else:
            try:
                encoder_dim = self.backbone_model.encoder.dim
            except Exception:
                import traceback
                logging.warning("Error getting encoder dim: " + traceback.format_exc())
        assert isinstance(encoder_dim, int) and encoder_dim > 0, \
            "Invalid encoder dim either inferred from model.encoder.dim or `hidden_size` provided"
        self.num_labels = num_labels
        self.hidden_size = encoder_dim
        self.train_path = train_path
        self.val_path = val_path
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.output_path = output_path
        self.verbose = verbose
        assert isinstance(self.output_path, str) and self.output_path.startswith('hdfs://')
        hmkdir(self.output_path)
        self.save_file_key = 'cmnli_linearprob'
        self.trainer_kwargs = trainer_kwargs if isinstance(trainer_kwargs, dict) else {}
        self.model_kwargs = model_kwargs if isinstance(model_kwargs, dict) else {}
        self.trainer_kwargs.update({
            'default_root_dir': 'benchmark_' + self.save_file_key,
            'default_hdfs_dir': '',
            'logger': 'console',
            'enable_versions': True,
            'max_epochs': self.trainer_kwargs.get('max_epochs', 10),
            'val_check_interval': -1,
            'enable_checkpoint': False,
            'accelerator_kwargs': {'enable_ddp': False},
            'log_every_n_steps': 2000,
            'precision': 16,
            'find_unused_parameters': False,
        })
        self.fit_batch_size = global_fit_batch_size
        self.train_embs = None
        self.train_labels = None
        self.val_embs = None
        self.val_labels = None

    def run(self):
        torch.cuda.empty_cache()
        self.encode_train()
        torch.cuda.empty_cache()
        self.encode_val()
        DIST_ENV.barrier()
        result = {}
        if DIST_ENV.rank == 0:
            result = self.fit_and_validate()
        torch.cuda.empty_cache()
        return result

    def emb_file_path(self, type_str):
        more = f'{self.save_file_key}_' if self.save_file_key else ''
        return os.path.join(self.output_path, f'{more}{type_str}.emb.%s')

    def save(self, embs, type_str, surfix=''):
        # write info
        surfix = str(DIST_ENV.rank) + surfix
        # write emb
        if embs is not None:
            save(embs, self.emb_file_path(type_str) % surfix)

    def encode_train(self, is_save=True):
        """Use all ranks to run inference, remove duplicate and save to hdfs"""
        train_loader = DistributedCruiseDataLoader(
            data_sources=[[self.train_path]],
            keys_or_columns=None,
            batch_sizes=[self.train_batch_size],
            num_workers=self.num_workers,
            num_readers=[1],
            decode_fn_list=None,
            processor=NLIProcessor(batch_size=self.train_batch_size),
            source_types=['parquet'],
            remain_sample_idx=False,
            drop_last=False,
            predefined_steps=-1,
            shuffle=False,
        )
        logging.info("[Rank %d/%d] CMNLI linear prob: total train batch: %d with bsz %d." % (
                DIST_ENV.rank, DIST_ENV.world_size, len(train_loader), self.train_batch_size))
        embs = []
        labels = []
        with torch.no_grad():
            for batch in train_loader:
                _num_valid_samples = batch.pop('_num_valid_samples', self.train_batch_size)
                label = batch.pop('labels')
                batch = move_data_to_device(batch, device=self.backbone_model.device)
                sequence_output = self.backbone_model(**batch)['sequence_output']
                enc_out = sequence_output[:_num_valid_samples, 0, :]
                label = label[:_num_valid_samples]
                embs.append(enc_out.cpu())
                labels.append(label.cpu())
        train_loader.terminate()
        embs = torch.cat(embs, dim=0)
        labels = torch.cat(labels, dim=0)
        logging.info(f'[Rank {DIST_ENV.rank}/{DIST_ENV.world_size}] CMNLI Linearprob: inferenced {embs.shape} train embs, labels: {labels.shape}')
        # local_train_steps = torch.tensor(int(np.ceil(embs.shape[0] / self.fit_batch_size))).cuda()
        # max_train_steps = torch.max(DIST_ENV.all_gather(local_train_steps)).item()
        # local_num_sample = max_train_steps * self.fit_batch_size
        # local_num_to_pad = local_num_sample - embs.shape[0]
        # assert local_num_to_pad >= 0
        # if local_num_to_pad:
        #     pad_indices = np.random.choice(np.arange(embs.shape[0]), local_num_to_pad)
        #     embs = torch.cat((embs, embs[pad_indices, :]), dim=0)
        #     labels = torch.cat((labels, labels[pad_indices]), dim=0)
        #     logging.info(f'[Rank {DIST_ENV.rank}/{DIST_ENV.world_size}] CMNLI Linearprob: padded to {embs.shape} train embs, labels: {labels.shape}')
        # self.train_embs = embs
        # self.train_labels = labels
        if is_save:
            self.save(embs, type_str='train')
            self.save(labels, type_str='trnlbl')

    def encode_val(self, is_save=True):
        """Use all ranks to run inference, remove duplicate and save to hdfs"""
        val_loader = DistributedCruiseDataLoader(
            data_sources=[[self.val_path]],
            keys_or_columns=None,
            batch_sizes=[self.val_batch_size],
            num_workers=self.num_workers,
            num_readers=[1],
            decode_fn_list=None,
            processor=NLIProcessor(batch_size=self.val_batch_size),
            source_types=['parquet'],
            remain_sample_idx=False,
            drop_last=False,
            predefined_steps=-1,
            shuffle=False,
        )
        logging.info("[Rank %d/%d] CMNLI linear prob: total val batch: %d with bsz %d." % (
                DIST_ENV.rank, DIST_ENV.world_size, len(val_loader), self.val_batch_size))
        embs = []
        labels = []
        with torch.no_grad():
            for batch in val_loader:
                _num_valid_samples = batch.pop('_num_valid_samples', self.val_batch_size)
                label = batch.pop('labels')
                batch = move_data_to_device(batch, device=self.backbone_model.device)
                sequence_output = self.backbone_model(**batch)['sequence_output']
                enc_out = sequence_output[:_num_valid_samples, 0, :]
                label = label[:_num_valid_samples]
                embs.append(enc_out.cpu())
                labels.append(label.cpu())
        val_loader.terminate()
        embs = torch.cat(embs, dim=0)
        labels = torch.cat(labels, dim=0)
        logging.info(f'[Rank {DIST_ENV.rank}/{DIST_ENV.world_size}] CMNLI Linearprob: inferenced {embs.shape} val embs, labels: {labels.shape}')
        # local_val_steps = torch.tensor(int(np.ceil(embs.shape[0] / self.fit_batch_size))).cuda()
        # max_val_steps = torch.max(DIST_ENV.all_gather(local_val_steps)).item()
        # local_num_sample = max_val_steps * self.fit_batch_size
        # local_num_to_pad = local_num_sample - embs.shape[0]
        # assert local_num_to_pad >= 0
        # if local_num_to_pad:
        #     pad_indices = np.random.choice(np.arange(embs.shape[0]), local_num_to_pad)
        #     embs = torch.cat((embs, embs[pad_indices, :]), dim=0)
        #     val_pad_labels = torch.ones_like(labels[pad_indices]) * -1
        #     labels = torch.cat((labels, val_pad_labels), dim=0)
        # logging.info(f'[Rank {DIST_ENV.rank}/{DIST_ENV.world_size}] CMNLI Linearprob: padded to {embs.shape} val embs, labels: {labels.shape}')
        # self.val_embs = embs
        # self.val_labels = labels
        if is_save:
            self.save(embs, type_str='val')
            self.save(labels, type_str='vlbl')

    def fit_and_validate(self):
        assert DIST_ENV.rank == 0
        hmkdir('./cache')
        self.model_kwargs.update({'num_labels': self.num_labels, 'hidden_size': self.hidden_size})
        fit_config = {
            'emb_file_path': [
                self.emb_file_path('train') % '*', self.emb_file_path('trnlbl') % '*',
                self.emb_file_path('val') % '*', self.emb_file_path('vlbl') % '*'],
            'batch_size': self.fit_batch_size,
            'trainer_kwargs': self.trainer_kwargs,
            'model_kwargs': self.model_kwargs,
            'result_path': os.path.abspath(os.path.join('./cache/', 'result.json'))
        }
        local_config = os.path.abspath(os.path.join('./cache/', 'config.json'))
        debug_log = os.path.abspath(os.path.join('./cache/', 'log.txt'))
        with open(local_config, 'w') as f:
            json.dump(fit_config, f)
        # launch fit/validate in other process
        try:
            os.remove(fit_config['result_path'])
        except FileNotFoundError:
            pass
        try:
            cruise_dir = os.path.abspath(os.path.join(os.path.dirname(cruise.__file__), '..'))
            export_path = f'PYTHONPATH={cruise_dir}:$PYTHONPATH'
            output = subprocess.check_output([f'{export_path} RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 LOCAL_WORLD_SIZE=1 python3 {os.path.abspath(__file__)} --config {local_config} 2>&1 | tee {debug_log}'], shell=True)
            if self.verbose:
                logging.info('Fit result:\n' + output.decode('utf-8') + '\n')
            with open(fit_config['result_path'], 'r') as fin:
                result = json.load(fin)
        except FileNotFoundError:
            try:
                with open(debug_log, 'r') as flog:
                    err_msg = flog.read()
            except FileNotFoundError:
                err_msg = ''
            logging.warning("Fit failed. Error log:\n" + err_msg)
            result = {}
        return result


def _fit_and_validate(config_file):
    logging.info('CMNLI Linear Probing: training classifier...')
    with open(config_file, 'r') as f:
        fit_config = json.load(f)
    # load train embedding
    emb_file_paths = fit_config['emb_file_path']
    tpaths = hglob(emb_file_paths[0])
    tlabels = hglob(emb_file_paths[1])
    tval_paths = hglob(emb_file_paths[2])
    tval_labels = hglob(emb_file_paths[3])
    train_embs = []
    train_labels = []
    val_embs = []
    val_labels = []
    for tp in sorted(tpaths):
        cur_temb = torch_io_load(tp)
        train_embs.append(cur_temb)
    train_embs = torch.cat(train_embs, dim=0)
    for tp in sorted(tlabels):
        cur_label = torch_io_load(tp)
        train_labels.append(cur_label)
    train_labels = torch.cat(train_labels, dim=0)
    for tp in sorted(tval_paths):
        cur_temb = torch_io_load(tp)
        val_embs.append(cur_temb)
    val_embs = torch.cat(val_embs, dim=0)
    for tp in sorted(tval_labels):
        cur_label = torch_io_load(tp)
        val_labels.append(cur_label)
    val_labels = torch.cat(val_labels, dim=0)
    logging.info('CMNLI Linear Probing: loaded train: %s, train label: %s, val: %s, val label: %s' % (
        train_embs.shape, train_labels.shape, val_embs.shape, val_labels.shape))
    # fixed configs for mlp training
    train_dataset = torch.utils.data.TensorDataset(train_embs.float(), train_labels.float())
    train_loader = DataLoader(
        train_dataset,
        batch_size=fit_config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True)
    val_dataset = torch.utils.data.TensorDataset(val_embs.float(), val_labels.float())
    val_loader = DataLoader(
        val_dataset,
        batch_size=fit_config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False)
    trainer = CruiseTrainer(**fit_config['trainer_kwargs'])
    model = NLIModel(**fit_config['model_kwargs'])
    trainer.fit(model, train_dataloader=train_loader)
    result = trainer.validate(model, val_dataloader=val_loader, verbose=False)
    with open(fit_config['result_path'], 'w') as f:
        json.dump(result, f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('subprocess fit')
    parser.add_argument('--config', required=True, type=str, help='config file')
    args = parser.parse_args()
    _fit_and_validate(args.config)
