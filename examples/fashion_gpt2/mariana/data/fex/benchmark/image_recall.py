"""Image recall benchmark, copied from fuxi benchmark, skip config loading"""
import os
import json
import logging

import torch

from cruise.utilities import DIST_ENV
from cruise.utilities.hdfs_io import hdfs_open, hglob, hopen, hmkdir
from cruise.utilities.cloud_io import load as torch_io_load
from cruise.utilities import move_data_to_device

from ..dataset.tusou_query import TusouQueryDataset
from ..dataset.tusou_image import TusouImageDataset


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


class ImageSearchRecallBenchmark:
    def __init__(self, model, vocab_file, output_path,
                 data_path="hdfs://haruna/home/byte_search_nlp_lq/user/huangwenguan/data/tusou/click_202057_cf_val",
                 query_batch_size=512,
                 image_batch_size=64,
                 data_size=70000,  # does not need to be accurate
                 *args, **kwargs):
        self.model = model
        self.vocab_file = vocab_file
        self.data_path = data_path
        self.output_path = output_path
        assert isinstance(self.output_path, str) and self.output_path.startswith('hdfs://')
        hmkdir(self.output_path)
        self.seq_len = 32  # fixed
        self.query_batch_size = query_batch_size
        self.image_batch_size = image_batch_size
        self.data_size = data_size
        self.save_file_key = 'tuso'

    @torch.no_grad()
    def run(self):
        torch.cuda.empty_cache()
        self.encode_query()
        torch.cuda.empty_cache()
        self.encode_image()
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

    def calc_recall_rate(self, topks=(1, 5, 10)):
        """
        先从文件里 load embedding，然后算 rate
        input: t_emb: [bsz, dim]; v_emb: [bsz, dim]
        output: t2i top@n; i2t top@n
        """

        logging.info('Tusou Recall: calculating recall ...')
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
                    query_map.append([jl['query'], jl['gid']])
                    queries.append(jl['query'])
        t_emb = torch.cat(t_emb, dim=0).half().cuda()
        logging.info('Tusou Recall: loaded query: %s, %s' % (len(queries), t_emb.shape))

        # load image embedding
        vpaths = hglob(self.emb_file_path('img') % '*')
        vinfo_paths = hglob(self.info_file_path('img') % '*')
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
        logging.info('Tusou Recall: loaded video: %s, %s' % (len(gids), v_emb.shape))

        # make ground truth
        vgt = []
        for g, q in gid_map:
            if q in queries:
                vgt.append(queries.index(q))
            else:  # 如果当前图片需要的query，不在已encode的query列表里，那ground truth是-1
                vgt.append(-1)

        tgt = []
        for q, g in query_map:
            if g in gids:
                tgt.append(gids.index(g))
            else:
                tgt.append(-1)
        # vgt = [queries.index(q) for _, q in gid_map]
        # tgt = [gids.index(g) for _, g in query_map]

        t_emb = torch.nn.functional.normalize(t_emb, dim=1)
        v_emb = torch.nn.functional.normalize(v_emb, dim=1)
        inter_mat = torch.mm(t_emb, v_emb.transpose(0, 1))
        result = {}
        s = '\n[Tuso Recall (zero shot)]\n'

        for topk in topks:
            _, rank_vt = inter_mat.topk(topk, dim=0)  # [img_size, topk]
            rank_vt = rank_vt.t()
            _, rank_tv = inter_mat.topk(topk, dim=1)  # [cap_size, topk]
            # logging.info('Tusou recall: vgt {} tgt {}'.format(vgt[:topk], tgt[:topk]))
            # logging.info(f'Tusou Recall: matrix shape {inter_mat.shape} {rank_vt.shape} {rank_tv.shape}')
            vcnt = 0.
            total = rank_vt.shape[0]
            for gt, vtopk in zip(vgt, rank_vt):
                if gt in vtopk.tolist():
                    vcnt += 1

            tcnt = 0.
            for gt, ttopk in zip(tgt, rank_tv):
                if gt in ttopk.tolist():
                    tcnt += 1

            s += 'V->T@%d: %s = %s/%s\n' % (topk, round(vcnt / total * 100, 4), vcnt, total)
            s += 'T->V@%d: %s = %s/%s\n' % (topk, round(tcnt / total * 100, 4), tcnt, total)
            result[f'tusou_v2t_{topk}'] = round(vcnt / total * 100, 4)
            result[f'tusou_t2v_{topk}'] = round(tcnt / total * 100, 4)

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
        rank0print(f'Tusou Recall: encoding query with bsz {batch_size} ...')
        dataset = TusouQueryDataset(self.vocab_file, self.data_path,
                                    shuffle=False,
                                    rank=DIST_ENV.rank,
                                    world_size=DIST_ENV.world_size,
                                    seq_len=self.seq_len,
                                    repeat=False, debug=True,
                                    data_size=self.data_size)
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
        data_iter = iter(loader)
        logging.info(f'[Rank {DIST_ENV.rank}/{DIST_ENV.world_size}] Tusou Recall: query dataload len: {loader_len}')
        while True:
            data, has_data, shoud_stop = get_data_all_rank(data_iter, self.model)
            # break if all ranks have no data to process
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
                logging.info(f'Tusou Recall: query embs batch step {i}/{loader_len}')
            i += 1

        t_embs = torch.cat(t_embs, dim=0)
        logging.info('[Rank %d/%d] Tusou Recall: encoded query %s queries, %s embs' % (
            DIST_ENV.rank, DIST_ENV.world_size, len(infos), t_embs.shape))
        if is_save:
            self.save(t_embs, infos, type_str='qry')

    def encode_image(self, is_save=True):
        batch_size = self.image_batch_size
        rank0print(f'Tusou Recall: encoding video with bsz {batch_size}...')
        if DIST_ENV.rank >= 128:
            logging.info("[Rank %d/%d] Tusou Recall: Skip due to oversize gpu worker.")
            return
        dataset = TusouImageDataset(self.data_path,
                                    shuffle=False,
                                    rank=DIST_ENV.rank,
                                    world_size=DIST_ENV.world_size,
                                    repeat=False,
                                    debug=True,
                                    data_size=self.data_size)
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=max(1, 128 // DIST_ENV.world_size),  # can handle up to 128 gpu
                                             collate_fn=dataset.collect_fn)

        v_embs = []
        infos = []
        device = torch.device(f'cuda:{DIST_ENV.local_rank}')

        loader_len = len(loader)
        data_iter = iter(loader)
        i = 0
        pre_data = pre_docs = None
        logging.info(f'[Rank {DIST_ENV.rank}/{DIST_ENV.world_size}] Tusou Recall: image dataloader len: {loader_len}')
        while True:
            data, has_data, shoud_stop = get_data_all_rank(data_iter, self.model)
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
            data['mode'] = 'v'
            if hasattr(self.model, 'module'):
                visual_cls = self.model.module.encode(**data)['pooled_out']  # [b, f, dim]
            else:
                visual_cls = self.model.encode(**data)['pooled_out']  # [b, f, dim]

            if has_data:
                v_embs.append(visual_cls.cpu())
                infos.extend(docs)
            if DIST_ENV.rank == 0 and i > 0 and i % 50 == 0:
                logging.info(f'Tusou Recall: image embs batch step {i}/{loader_len}')
            i += 1

        v_embs = torch.cat(v_embs, dim=0)
        logging.info(f'[Rank {DIST_ENV.rank}/{DIST_ENV.world_size}] Tusou Recall: encoded {len(infos)} images, {v_embs.shape} embs')
        if is_save:
            self.save(v_embs, infos, type_str='img')
