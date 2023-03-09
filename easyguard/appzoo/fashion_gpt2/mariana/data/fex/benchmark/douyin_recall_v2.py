"""Douyin recall benchmark v2, copied from fuxi benchmark, skip config loading"""
import os
import json
import random
import logging

import torch

from cruise.utilities import DIST_ENV
from cruise.utilities.hdfs_io import hdfs_open, hglob, hopen, hmkdir
from cruise.utilities.cloud_io import load as torch_io_load
from cruise.utilities import move_data_to_device

from ..dataset.douyin_query import DouyinQueryDataset
from ..dataset.douyin_frame_v2 import DouyinFrameDatasetV2
from ..dali_pipeline_f import ValFrameDecoderQueuePipeline
from ..dali_iter import PytorchDaliIter


def rank0print(str):
    if DIST_ENV.rank != 0:
        return
    print(str)


def save(obj, filepath: str, **kwargs):
    """ save model """
    if filepath.startswith("hdfs://"):
        with hopen(filepath, "wb") as writer:
            torch.save(obj, writer, **kwargs)
    else:
        torch.save(obj, filepath, **kwargs)


def get_data_all_rank(data_loader, dist):
    data = next(data_loader, None)
    has_data = torch.tensor(0) if data is None else torch.tensor(1)
    has_data_max = torch.max(dist.all_gather(has_data)).item()
    return data, has_data.item() == 1, has_data_max == 0


class DouyinSearchRecallBenchmarkV2:
    def __init__(self, model, vocab_file, output_path,
                 query_path="hdfs://haruna/home/byte_search_nlp_lq/user/huangwenguan/data/bench/dy_label_recall_0508/recall.q",
                 video_path="hdfs://haruna/home/byte_search_nlp_lq/user/huangwenguan/data/bench/dy_label_recall_0508/recall.f",
                 mode='v',
                 is_bert_style=True, need_text=True, need_ocr=False,
                 query_batch_size=512,
                 video_batch_size=64,
                 data_size=300000,
                 *args, **kwargs):
        self.model = model
        self.vocab_file = vocab_file
        self.query_path = query_path
        self.video_path = video_path
        self.output_path = output_path
        assert isinstance(self.output_path, str) and self.output_path.startswith('hdfs://')
        hmkdir(self.output_path)
        self.mode = mode
        self.is_bert_style = is_bert_style
        self.need_text = need_text
        self.need_ocr = need_ocr
        self.seq_len = 32  # fixed
        self.query_batch_size = query_batch_size
        self.video_batch_size = video_batch_size
        self.data_size = data_size
        self.save_file_key = 'dyv2'

    @torch.no_grad()
    def run(self):
        torch.cuda.empty_cache()
        self.encode_video()
        torch.cuda.empty_cache()
        self.encode_query()
        DIST_ENV.barrier()
        result = {}
        if DIST_ENV.rank == 0:
            result = self.calc_recall_rate()
        torch.cuda.empty_cache()
        return result

    def info_file_path(self, type_str):
        more = f'{self.save_file_key}_' if self.save_file_key else ''
        return os.path.join(self.output_path, f'{more}{type_str}.info.%s')

    def emb_file_path(self, type_str):
        more = f'{self.save_file_key}_' if self.save_file_key else ''
        return os.path.join(self.output_path, f'{more}{type_str}.emb.%s')

    def save(self, embs, infos, type_str, surfix=''):
        # write info
        surfix = str(DIST_ENV.rank) + surfix
        if infos is not None:
            with hdfs_open(self.info_file_path(type_str) % surfix, 'w') as f:
                for info in infos:
                    towrite_str = json.dumps(info, ensure_ascii=False) + '\n'
                    if self.output_path.startswith('hdfs'):
                        towrite_str = towrite_str.encode()
                    f.write(towrite_str)
        # write emb
        if embs is not None:
            save(embs, self.emb_file_path(type_str) % surfix)

    def calc_recall_rate(self, topks=(10, 100, 1000)):
        """
        先从文件里 load embedding，然后算 rate
        input: t_emb: [bsz, dim]; v_emb: [bsz, dim]
        output: t2i top@n; i2t top@n
        """

        logging.info('Douyin Search Recall V2: calculating recall ...')
        # load query embedding
        tpaths = hglob(self.emb_file_path('qry') % '*')
        tinfo_paths = hglob(self.info_file_path('qry') % '*')
        t_emb = []
        query_map = []
        queries = []
        for tp, tip in zip(sorted(tpaths), sorted(tinfo_paths)):
            cur_temb = torch_io_load(tp)
            t_emb.append(cur_temb)
            with hdfs_open(tip) as f:
                for i, l in enumerate(f):
                    jl = json.loads(l)
                    query_map.append([jl['query'], jl['gids']])
                    queries.append(jl['query'])
        t_emb = torch.cat(t_emb, dim=0).half().cuda()
        logging.info('Douyin Search Recall V2: loaded query: %s, %s' % (len(queries), t_emb.shape))

        # load image embedding
        vpaths = hglob(self.emb_file_path('frm') % '*')
        vinfo_paths = hglob(self.info_file_path('frm') % '*')
        v_emb = []
        gids = []
        for tp, tip in zip(sorted(vpaths), sorted(vinfo_paths)):
            cur_temb = torch_io_load(tp)
            v_emb.append(cur_temb)
            with hdfs_open(tip) as f:
                for i, l in enumerate(f):
                    jl = json.loads(l)
                    gids.append(jl['gid'])
        v_emb = torch.cat(v_emb, dim=0).half().cuda()
        logging.info('Douyin Search Recall V2: loaded video: %s, %s' % (len(gids), v_emb.shape))

        t_emb = torch.nn.functional.normalize(t_emb, dim=1)
        v_emb = torch.nn.functional.normalize(v_emb, dim=1)
        inter_mat = torch.mm(t_emb, v_emb.transpose(0, 1))
        result = {}
        s = '\n[Douyin Recall V2 (zero shot)]\n'
        for topk in topks:
            _, rank_tv = inter_mat.topk(topk, dim=1)  # [query_size, topk]
            logging.info(f'Douyin Search Recall V2: matrix shape {inter_mat.shape} {rank_tv.shape}')

            t_correct = 0.
            t_total = rank_tv.shape[0]
            doc_correct = 0.
            doc_total = 0
            for i, ttopk in enumerate(rank_tv):
                cur_query, ground_truth_gids = query_map[i]
                ttopk_gids = [gids[idx] for idx in ttopk.tolist()]
                if random.random() < 0.001:
                    logging.debug(f'Douyin Search Recall V2: query: {cur_query}')
                    logging.debug(f'Douyin Search Recall V2: ground truth: {ground_truth_gids}')
                    logging.debug(f'Douyin Search Recall V2: top k: {ttopk_gids}')
                gt_hit_num = sum([g in ttopk_gids for g in ground_truth_gids])
                # query level 的召回率：只要召回的doc里有一个在ground truth 里，都认为这个query是对的
                if gt_hit_num > 0:
                    t_correct += 1
                # doc level 的召回率：就看每个doc是否被召回了
                doc_total += len(ground_truth_gids)
                doc_correct += gt_hit_num

            s += 'Query @%s: %s = %s/%s\n' % (
                topk, round(t_correct / t_total * 100, 4), t_correct, t_total)
            s += 'Doc @%s: %s = %s/%s\n' % (
                topk, round(doc_correct / doc_total * 100,
                            4), doc_correct, doc_total)
            result[f'douyin_recall_v2_query_{topk}'] = round(t_correct / t_total * 100, 4)
            result[f'douyin_recall_v2_doc_{topk}'] = round(doc_correct / doc_total * 100, 4)
        logging.info(s)
        with hdfs_open(os.path.join(self.output_path, 'result.txt.%s' % DIST_ENV.rank), 'a') as resfn:
            if self.output_path.startswith('hdfs'):
                s = s.encode()
                resfn.write(s)
        return result

    def encode_query(self, is_save=True):
        """
        对query 进行编码
        因为query是作为g
        """
        batch_size = self.query_batch_size
        rank0print(f'Douyin Search Recall V2: encoding query with bsz {batch_size} ...')
        dataset = DouyinQueryDataset(self.vocab_file, self.query_path,
                                     shuffle=False,
                                     rank=DIST_ENV.rank,
                                     world_size=DIST_ENV.world_size,
                                     seq_len=self.seq_len,
                                     repeat=False, debug=True)
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=2,
                                             collate_fn=dataset.collect_fn)
        t_embs = []
        infos = []
        device = torch.device(f'cuda:{DIST_ENV.local_rank}')

        i = 0
        pre_data = pre_docs = None
        loader_len = len(loader)
        data_iter=iter(loader)
        logging.info(f'[Rank {DIST_ENV.rank}/{DIST_ENV.world_size}] Douyin Search Recall V2: query dataload len: {loader_len}')
        while True:
            data, has_data, shoud_stop = get_data_all_rank(data_iter, self.model)
            #break if all ranks have no data to process
            if shoud_stop:
                break
            if has_data:
                pre_data = data
                docs = data.pop('docs', None)
                pre_docs = docs

            else:
                pre_data.pop('mode', None)
                data = pre_data
                docs = pre_docs

            data = move_data_to_device(data, device, non_blocking=False)
            data['mode'] = 't'
            if hasattr(self.model, 'module'):
                text_cls = self.model.module.encode(**data)['pooled_out']
            else:
                text_cls = self.model.encode(**data)['pooled_out']

            if has_data:
                t_embs.append(text_cls.cpu())
                infos.extend(docs)
            if DIST_ENV.rank == 0 and i > 0 and i % 50 == 0:
                logging.info(f'Douyin Search Recall V2: query embs batch step {i}/{loader_len}')
            i += 1

        t_embs = torch.cat(t_embs, dim=0)
        logging.info('[Rank %d/%d] Douyin Search Recall V2: encoded query %s queries, %s embs' % (
            DIST_ENV.rank, DIST_ENV.world_size, len(infos), t_embs.shape))
        if is_save:
            self.save(t_embs, infos, type_str='qry')

    def encode_video(self, is_save=True):
        batch_size = self.video_batch_size
        rank0print(f'Douyin Search Recall V2: encoding video with bsz {batch_size}...')
        if DIST_ENV.rank >= 128:
            logging.info("[Rank %d/%d] Douyin Search Recall V2: Skip due to oversize gpu worker." % (
                DIST_ENV.rank, DIST_ENV.world_size))
            return
        dataset = DouyinFrameDatasetV2(self.vocab_file, self.video_path,
                                       shuffle=False,
                                       rank=DIST_ENV.rank,
                                       world_size=DIST_ENV.world_size,
                                       seq_len=self.seq_len,
                                       is_bert_style=self.is_bert_style,
                                       need_text=self.need_text,
                                       need_ocr=self.need_ocr,
                                       repeat=False,
                                       debug=True,
                                       data_size=self.data_size)
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=max(1, 128 // DIST_ENV.world_size),  # can handle up to 128 gpu
                                             collate_fn=dataset.collect_fn)

        val_pipeline = ValFrameDecoderQueuePipeline(batch_size=batch_size,
                                                    num_threads=2,
                                                    device_id=DIST_ENV.local_rank,
                                                    external_data=loader,
                                                    min_size=224,
                                                    max_size=224)
        from nvidia.dali.plugin.pytorch import LastBatchPolicy
        val_dali_iter = PytorchDaliIter(dali_pipeline=val_pipeline,
                                        size=len(loader),
                                        output_map=['image'],
                                        last_batch_policy=LastBatchPolicy.FILL,
                                        last_batch_padded=True,
                                        dynamic_shape=True)

        v_embs = []
        infos = []
        device = torch.device(f'cuda:{DIST_ENV.local_rank}')

        loader_len = len(val_dali_iter)
        i = 0
        pre_data = pre_docs = None
        logging.info(f'[Rank {DIST_ENV.rank}/{DIST_ENV.world_size}] Douyin Search Recall V2: video dataload len: {loader_len}')
        while True:
            data, has_data, shoud_stop = get_data_all_rank(val_dali_iter, self.model)
            if shoud_stop:
                break
            if has_data:
                pre_data = data
                docs = data.pop('docs', None)
                pre_docs = docs

            else:
                pre_data.pop('mode', None)
                data = pre_data
                docs = pre_docs

            data = move_data_to_device(data, device, non_blocking=False)
            data['mode'] = self.mode
            if hasattr(self.model, 'module'):
                visual_cls = self.model.module.encode(**data)['pooled_out']  # [b, f, dim]
            else:
                visual_cls = self.model.encode(**data)['pooled_out']  # [b, f, dim]
            visual_cls = visual_cls.mean(1)
            if len(docs) != len(visual_cls):
                logging.info(
                    f'[Rank {DIST_ENV.rank}/{DIST_ENV.world_size}] Douyin Search Recall V2: encounter bad batch: {i} {len(docs)} {len(visual_cls)}')
                visual_cls = visual_cls[:len(docs)]

            if has_data:
                v_embs.append(visual_cls.cpu())
                infos.extend(docs)
            if DIST_ENV.rank == 0 and i > 0 and i % 50 == 0:
                logging.info(f'Douyin Search Recall V2: video embs batch step {i}/{loader_len}')
            i += 1

        v_embs = torch.cat(v_embs, dim=0)
        logging.info(f'[Rank {DIST_ENV.rank}/{DIST_ENV.world_size}] Douyin Search Recall V2: encoded {len(infos)} videos, {v_embs.shape} embs')
        if is_save:
            self.save(v_embs, infos, type_str='frm')
