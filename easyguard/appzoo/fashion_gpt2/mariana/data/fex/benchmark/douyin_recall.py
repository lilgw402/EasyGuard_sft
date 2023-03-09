"""Douyin recall benchmark v1, copied from fuxi benchmark, skip config loading"""
import os
import json
import logging

import torch

from cruise.utilities.distributed import DIST_ENV
from cruise.utilities.hdfs_io import hdfs_open, hglob, hopen, hmkdir
from cruise.utilities.cloud_io import load as torch_io_load


from ..dataset.douyin_query import DouyinQueryDataset
from ..dataset.douyin_frame import DouyinFrameDataset


def rank0print(str):
    if DIST_ENV.rank != 0:
        return
    logging.info(str)


def save(obj, filepath: str, **kwargs):
    """ save model """
    if filepath.startswith("hdfs://"):
        with hopen(filepath, "wb") as writer:
            torch.save(obj, writer, **kwargs)
    else:
        torch.save(obj, filepath, **kwargs)


class DouyinSearchRecallBenchmark:
    def __init__(self, model, vocab_file, output_path,
                 query_path="hdfs://haruna/home/byte_search_nlp_lq/user/huangwenguan/data/bench/dy_label_recall/recall.q",
                 video_path="hdfs://haruna/home/byte_search_nlp_lq/user/huangwenguan/data/bench/dy_label_recall/recall.f",
                 mode='v',
                 is_bert_style=True, need_text=True, need_ocr=False,
                 query_batch_size=512,
                 video_batch_size=64,
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
        self.save_file_key = 'dy'

    @torch.no_grad()
    def run(self):
        torch.cuda.empty_cache()
        self.encode_video()
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

    def calc_recall_rate(self, topk=(1, 5, 10,)):
        """
        先从文件里 load embedding，然后算 rate
        input: t_emb: [bsz, dim]; v_emb: [bsz, dim]
        output: t2i top@n; i2t top@n
        """

        logging.info('Douyin Search Recall V1: calculating recall ...')
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
        logging.info('Douyin Search Recall V1: loaded query: %s, %s' % (len(queries), t_emb.shape))

        # load image embedding
        vpaths = hglob(self.emb_file_path('frm') % '*')
        vinfo_paths = hglob(self.info_file_path('frm') % '*')
        v_emb = []
        gid_map = []
        gids = []
        for tp, tip in zip(sorted(vpaths), sorted(vinfo_paths)):
            cur_temb = torch_io_load(tp)
            v_emb.append(cur_temb)
            with hdfs_open(tip) as f:
                for i, l in enumerate(f):
                    jl = json.loads(l)
                    gid_map.append([jl['gid'], jl['query']])
                    gids.append(jl['gid'])
        v_emb = torch.cat(v_emb, dim=0).half().cuda()
        logging.info('Douyin Search Recall V1: loaded video: %s, %s' % (len(gids), v_emb.shape))

        # make ground truth

        vgt = []
        for g, q in gid_map:
            if q in queries:
                vgt.append(queries.index(q))
            else:  # 如果当前图片需要的query，不在已encode的query列表里，那ground truth是-1
                vgt.append(-1)

        tgt = []
        for q, gs in query_map:
            cur_pac = []
            for g in gs:
                if g in gids:
                    cur_pac.append(gids.index(g))
                else:
                    cur_pac.append(-1)
            tgt.append(cur_pac)
        # vgt = [queries.index(q) for _, q in gid_map]
        # tgt = [gids.index(g) for _, g in query_map]
        logging.info('Douyin Search Recall V1: gt: {vgt[:3]}, {tgt[:3]}')

        t_emb = torch.nn.functional.normalize(t_emb, dim=1)
        v_emb = torch.nn.functional.normalize(v_emb, dim=1)
        inter_mat = torch.mm(t_emb, v_emb.transpose(0, 1))
        # batch_size = t_emb.size(0)

        _, rank_vt = inter_mat.topk(10, dim=0)  # [img_size, 10]
        rank_vt = rank_vt.t()
        _, rank_tv = inter_mat.topk(10, dim=1)  # [cap_size, 10]
        logging.info(f'Douyin Search Recall V1: matrix shape {inter_mat.shape} {rank_vt.shape} {rank_tv.shape}')
        result = {}
        with hdfs_open(os.path.join(self.output_path, 'result.txt.%s' % DIST_ENV.rank), 'a') as resfn:
            vcnt = 0.
            total = rank_vt.shape[0]
            for gt, vtopk in zip(vgt, rank_vt):
                if gt in vtopk.tolist():
                    vcnt += 1

            tcnt = 0.
            t_total = rank_tv.shape[0]
            for gt, ttopk in zip(tgt, rank_tv):
                if any([g in ttopk.tolist() for g in gt]):
                    tcnt += 1

            s = '\n[Douyin Recall (zero shot)]\n'
            s += 'V->T@10: %s = %s/%s\n' % (round(vcnt / total * 100, 4), vcnt, total)
            s += 'T->V@10: %s = %s/%s' % (round(tcnt / t_total * 100, 4), tcnt, t_total)
            logging.info(s)
            if self.output_path.startswith('hdfs'):
                s = s.encode()
                resfn.write(s)
            result = {'douyin_recall_v2t_10': round(vcnt / total * 100, 4), 'douyin_recall_t2v_10': round(tcnt / t_total * 100, 4)}
        return result

    def encode_query(self, is_save=True):
        """
        对query 进行编码
        因为query是作为g
        """
        batch_size = self.query_batch_size
        rank0print(f'Douyin Search Recall V1: encoding query with bsz {batch_size} ...')
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
        for i, data in enumerate(loader):
            docs = data.pop('docs')
            for k, v in data.items():
                data[k] = v.cuda(non_blocking=True)
            data['mode'] = 't'
            if hasattr(self.model, 'module'):
                text_cls = self.model.module.encode(**data)['pooled_out']
            else:
                text_cls = self.model.encode(**data)['pooled_out']
            t_embs.append(text_cls.cpu())
            infos.extend(docs)
        t_embs = torch.cat(t_embs, dim=0)
        logging.info('[Rank %d/%d] Douyin Search Recall V1: encoded %s queries, %s embs' % (
            DIST_ENV.rank, DIST_ENV.world_size, len(infos), t_embs.shape))
        if is_save:
            self.save(t_embs, infos, type_str='qry')

    def encode_video(self, is_save=True):
        batch_size = self.video_batch_size
        rank0print(f'Douyin Search Recall V1: encoding video with bsz {batch_size}...')
        if DIST_ENV.rank >= 128:
            logging.info("[Rank %d/%d] Douyin Search Recall V1: Skip due to oversize gpu worker.")
            return
        dataset = DouyinFrameDataset(self.vocab_file, self.video_path,
                                     shuffle=False,
                                     rank=DIST_ENV.rank,
                                     world_size=DIST_ENV.world_size,
                                     seq_len=self.seq_len,
                                     is_bert_style=self.is_bert_style,
                                     need_text=self.need_text,
                                     need_ocr=self.need_ocr,
                                     repeat=False,
                                     debug=True)
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=max(1, 128 // DIST_ENV.world_size),  # can handle up to 128 gpu
                                             collate_fn=dataset.collect_fn)
        v_embs = []
        infos = []
        for i, data in enumerate(loader):
            docs = data.pop('docs')
            for k, v in data.items():
                data[k] = v.cuda(non_blocking=True)
            data['mode'] = self.mode
            if hasattr(self.model, 'module'):
                visual_cls = self.model.module.encode(**data)['pooled_out']  # [b, f, dim]
            else:
                visual_cls = self.model.encode(**data)['pooled_out']  # [b, f, dim]
            visual_cls = visual_cls.mean(1)
            v_embs.append(visual_cls.cpu())
            infos.extend(docs)
        v_embs = torch.cat(v_embs, dim=0)
        logging.info('[Rank %d/%d] Douyin Search Recall V1: encoded %s videos, %s embs' % (
            DIST_ENV.rank, DIST_ENV.world_size, len(infos), v_embs.shape))
        if is_save:
            self.save(v_embs, infos, type_str='frm')
