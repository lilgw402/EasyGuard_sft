import os
import sys
import site
import importlib
import subprocess
import torch
from pathlib import Path
from importlib import reload
from packaging import version
from titan.utils.misc import download_and_extract_scm_permission, download_http_to_local_file
from titan.utils.hdfs import download_from_hdfs
from titan.utils.logger import logger
from torch.distributed import is_initialized, barrier
from torch import nn


if os.getenv("ARNOLD_TRIAL_ID", None):
    SAVE_PREFIX = "/opt/tiger"
else:
    SAVE_PREFIX = "/tmp"


chinese_model_hdfs = {
    'zh_albert_base_l6_mix_oldcut_20200315_20200315': 'hdfs://haruna/nlp/public/models/hub/pretrained/zh_albert_base_l6_mix_oldcut_20200315_20200315',
    'zh_albert_base_l6_qa_site_web_l1_64gpus_20200520': 'hdfs://haruna/nlp/public/models/hub/pretrained/zh_albert_base_l6_qa_site_web_l1_64gpus_20200520',
    'zh_electra_base_6l_share_qa_site_web_l1_64gpus_all_fp16_20200526': 'hdfs://haruna/nlp/public/models/hub/pretrained/zh_electra_base_6l_share_qa_site_web_l1_64gpus_all_fp16_20200526',
    'zh_deberta_base_l6_emd_20210720': 'hdfs://haruna/nlp/public/models/hub/pretrained/zh_deberta_base_l6_emd_20210720',
    'zh_deberta_moe': 'hdfs://haruna/nlp/lixiang.0310/train/6450431/model_state_epoch_0_batch_1000000-1000000.th'
}
LibCutDir = os.path.join(SAVE_PREFIX, 'libcut_data_zh')


# PTX tokenizer has text cutter, so we add a wrapper here
class TokenizerWrapper:
    def __init__(self, cutter, ptx_tokenizer, vocab_path, pad_idx=2) -> None:
        self.cutter = cutter
        self.tokenizer = ptx_tokenizer
        self.vocab = {}
        self.pad_idx = pad_idx
        with open(vocab_path) as fi:
            for index, line in enumerate(fi):
                self.vocab[line[:-1]] = index
    
    def process_text(self, text: str, max_len: int, strict=False):
        words = self.cutter(text, 'FINE')
        tokens = self.tokenizer(words)
        token_ids = list()
        for token in tokens:
            for tk in token:
                token_ids.append(self.vocab.get(tk, self.vocab['[UNK]']))
        # token_ids = [self.vocab.get(token, self.vocab['[UNK]']) for token in tokens]
        token_ids = token_ids[: max_len]
        if strict and len(token_ids) < max_len:
            token_ids += [self.pad_idx] * (max_len - len(token_ids))
        return token_ids


class MoEDeBERTaWrapper(nn.Module):
    def __init__(self, config_path, model_path):
        super().__init__()
        from ptx.model import Model
        from ptx.registry import Option
        self.option = Option.from_file(config_path)
        self.bert = Model.from_option(self.option)
        state_dict = {k: v for k, v in torch.load(
            model_path).items() if not k.startswith('cls.')}
        self.bert.load_state_dict(state_dict, strict=False)
    
    def forward(self, input_ids, segment_ids, **kwargs):
        return self.bert(
            input_ids=input_ids,
            segment_ids=segment_ids,
            output_pooled=True
        )


def download_ptx(ptx_dest):
    if not Path(ptx_dest).exists():
        # Download SCM
        download_and_extract_scm_permission(ptx_dest, 'nlp.lib.ptx2', '1.0.0.329')


def hdfs_download(src_path, dst_path):
    if os.path.isdir(dst_path):
        basename = os.path.basename(src_path)
        dst_path = os.path.join(dst_path, basename)
    if os.path.exists(dst_path):
        pass
    else:
        download_from_hdfs(src_path, dst_path)


def scm_download(local_path, scm, version):
    if not os.path.exists(local_path):
        download_and_extract_scm_permission(local_path, scm, version)


def module_exist(module_name):
    return importlib.util.find_spec(module_name)


def create_ptx_chinese_model(model_name, local_rank=0):
    assert model_name in chinese_model_hdfs, f'ptx model name error, please select one from {chinese_model_hdfs.keys()}'
    exist_ptx = any(path.endswith('ptx') for path in sys.path)
    local_dir_path = os.path.join(SAVE_PREFIX, model_name)
    ptx_dest = os.path.join(SAVE_PREFIX, 'ptx')
    if not exist_ptx and local_rank == 0:
        download_ptx(ptx_dest)
    if local_rank == 0:
        hdfs_download(chinese_model_hdfs[model_name], local_dir_path)
    sync()
    if not exist_ptx:
        sys.path.insert(0, ptx_dest)
    
    # Begin to load model
    if 'electra' in model_name:
        import ptx.model.electra
        from ptx.model import Model
        m = Model.from_option(f'file:{local_dir_path}|weights_file=$/best.pt')
    elif model_name == 'zh_deberta_moe':
        deberta_doct_dir = os.path.join(SAVE_PREFIX, 'deberta_moe')
        if not os.path.exists(deberta_doct_dir) and local_rank == 0:
            install_deberta_moe_dependencies(deberta_doct_dir)
            import shutil
            shutil.move(local_dir_path, os.path.join(deberta_doct_dir, 'model.pt'))
        sync()
        reload(site)
        sys.path.insert(0, deberta_doct_dir)
        # this is needed for model definition is registered to PTX in doct module
        import doct
        from ptx.model import Model
        from ptx.registry import Option
        option = Option.from_file(os.path.join(deberta_doct_dir, 'task/distill/bert_for_cmnli_tokenization.jsonnet'))
        option = substitute_deberta_option_path(option, deberta_doct_dir)
        model = MoEDeBERTaWrapper(option['model']['config_path'], option['model']['model_path'])
        return model
    else:
        from ptx.model import Model
        m = Model.from_option(f'file:{local_dir_path}')
    return m


def substitute_deberta_option_path(option, doct_path):
    option['train_data_path'] = os.path.join(doct_path, option['train_data_path'])
    option['validation_data_path'] = os.path.join(doct_path, option['validation_data_path'])
    option['data_parser']['vocab_path'] = os.path.join(doct_path, option['data_parser']['vocab_path'])
    option['data_parser']['libcut_path'] = os.path.join(doct_path, option['data_parser']['libcut_path'])
    option['model']['config_path'] = os.path.join(doct_path, option['model']['config_path'])
    option['model']['model_path'] = os.path.join(doct_path, option['model']['model_path'])
    option['workdir'] = os.path.join(doct_path, option['workdir'])
    option['serialization_dir'] = os.path.join(doct_path, option['serialization_dir'])
    return option


def sync():
    if is_initialized():
        barrier()


def tokenizer_dependencies_installed():
    text_tokenizer = any('text_tokenizer' in path for path in sys.path)
    matx = any('matx' in path for path in sys.path)
    text_cutter = any('text_cutter' in path for path in sys.path)
    return text_tokenizer and matx and text_cutter


def check_package_missing(module_name):
    return any(module_name in path for path in sys.path)


def create_ptx_chinese_tokenizer(model_name, local_rank=0):
    python = sys.executable
    assert model_name in chinese_model_hdfs, f'ptx model name error, please select one from {chinese_model_hdfs.keys()}'
    # download matx for loading pipeline
    missing = []
    if not module_exist('text_cutter'):
        missing.append('http://d.scm.byted.org/api/v2/download/ceph:search.nlp.libcut_py_2.3.0.35.tar.gz')
    if not module_exist('matx'):
        missing.append('http://d.scm.byted.org/api/v2/download/ceph:byted.matx.matx4_pip_wheels_1.4.2.4.tar.gz')
    if not module_exist('text_tokenizer'):
        missing.append('http://d.scm.byted.org/api/v2/download/ceph:nlp.tokenizer.py_1.0.0.76.tar.gz')
    if missing and local_rank == 0:
        subprocess.check_call([python, '-m', 'pip', 'install', '-i', 'https://bytedpypi.byted.org/simple/', '--no-cache-dir', *missing], stdout=subprocess.DEVNULL)
    if 'deberta' in model_name and local_rank == 0:
        scm_download(os.path.join(SAVE_PREFIX, 'libcut_data_zh_20200827fix2'), 'search.nlp.libcut_data_zh_20200827fix2', '1.0.0.2')
    elif not os.path.exists(LibCutDir) and local_rank == 0:
        scm_download(LibCutDir, 'toutiao.nlp.libcut_data_zh', '1.0.0.1')
    
    exist_ptx = any(path.endswith('ptx') for path in sys.path) or module_exist('ptx')
    ptx_install = os.path.join(SAVE_PREFIX, 'ptx')
    if not exist_ptx:
        if not os.path.exists(ptx_install):
            if local_rank == 0:
                download_ptx(ptx_install)
    
    sync()
    reload(site)
    sys.path.insert(0, ptx_install)

    if model_name == 'zh_deberta_moe':
        import matx
        from ptx.registry import Option
        deberta_doct_dir = os.path.join(SAVE_PREFIX, 'deberta_moe')
        if local_rank == 0:
            # download tokenizer
            hdfs_download('hdfs://haruna/nlp/wuwei/models/zh/ptx_models/tokenization/newcut_0827_vocab.json', os.path.join(deberta_doct_dir, 'vocab.txt'))
            hdfs_download('hdfs://haruna/nlp/wuwei/models/zh/ptx_models/tokenization/libcut_data_zh_20200827', os.path.join(deberta_doct_dir, 'libcut_folder'))
            missing_pkg = []
            if not module_exist('matx'):
                missing_pkg.append('http://d.scm.byted.org/api/v2/download/ceph:byted.matx.matx4_pip_wheels_1.4.2.4.tar.gz')
            if not module_exist('matx_text'):
                missing_pkg.append('http://d.scm.byted.org/api/v2/download/ceph:byted.matx.matx_text_pip_wheels_1.4.2.3.tar.gz')
            if missing_pkg:
                subprocess.check_call([python, '-m', 'pip', 'install', '--no-cache-dir', *missing_pkg], stdout=subprocess.DEVNULL)
        sync()
        reload(site)
        option = Option.from_file(os.path.join(deberta_doct_dir, 'task/distill/bert_for_cmnli_tokenization.jsonnet'))
        option = substitute_deberta_option_path(option, deberta_doct_dir)
        vocab_path = option['data_parser']['vocab_path']
        libcut_path = option['data_parser']['libcut_path']
        if version.parse(matx.__version__) >= version.parse('1.6.0'):
            # https://bytedance.feishu.cn/docx/doxcnJafmnphVKQEoekNuB5UG5c
            from text_cutter import Cutter
            from text_tokenizer import WordPieceTokenizerOp
            ptx_tokenizer = matx.script(WordPieceTokenizerOp)(
                vocab_path=vocab_path, do_wordpiece=True)
            ptx_cutter = matx.script(Cutter)(option['data_parser']['cut_type'], libcut_path)
        else:
            from matx_text import WordPieceTokenizerOp, LibcutOp
            ptx_tokenizer = WordPieceTokenizerOp(location=vocab_path, do_wordpiece=True)
            ptx_cutter = LibcutOp(location=libcut_path, cut_type=option['data_parser']['cut_type'])
        return TokenizerWrapper(ptx_cutter, ptx_tokenizer, vocab_path)

    from ptx.matx.pipeline import Pipeline
    local_dir_path = os.path.join(SAVE_PREFIX, model_name)
    p = Pipeline.from_option(f'file:{local_dir_path}')
    return p


def install_deberta_moe_dependencies(deberta_doct_dir):
    # download doct from tos
    # doct is business code from the search team
    import time
    import tarfile
    download_path = deberta_doct_dir + str(time.time())
    download_http_to_local_file('https://tosv.byted.org/obj/nlp-model-cn/ebd6de49-d75d-47ea-87fe-0e60eb79e687', download_path)
    with tarfile.open(download_path) as tar:
        tar.extractall(deberta_doct_dir)
        files = os.listdir(deberta_doct_dir)
        s = str(time.time())
        os.rename(os.path.join(deberta_doct_dir, files[0]), deberta_doct_dir + '.' + s)
        os.rmdir(deberta_doct_dir)
        os.rename(deberta_doct_dir + '.' + s, deberta_doct_dir)
    os.remove(download_path)
    sys.path.insert(0, deberta_doct_dir)
    # hdfs_download('hdfs://haruna/nlp/wuwei/data/cmnli', deberta_doct_dir)

    # install janus and other pip package
    python = sys.executable
    pkgs = []
    if not module_exist('janus'):
        pkgs.append('byted-janus')
    if not module_exist('matx'):
        pkgs.append('http://d.scm.byted.org/api/v2/download/ceph:byted.matx.matx4_pip_wheels_1.4.2.4.tar.gz')
    if not module_exist('matx_text'):
        pkgs.append('http://d.scm.byted.org/api/v2/download/ceph:byted.matx.matx_text_pip_wheels_1.4.2.3.tar.gz')
    subprocess.check_call([python, '-m', 'pip', 'install', '--no-cache-dir', *pkgs, 'jsonnet', '-i', 'https://bytedpypi.byted.org/simple'], stdout=subprocess.DEVNULL)
