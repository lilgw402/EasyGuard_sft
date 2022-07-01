
from typing import List, Tuple

import os
import base64
import json
from absl import app
from absl import flags

import torch
from torch import nn
import matx
import matx_pytorch
import matx_text

from libcut_data_zh_20200827 import data_path as libcut_data_path

from fex.trace.model_convert import ModelConvertToFTAndFP16
from fex.utils.load import load_from_pretrain
from fex.utils.hdfs_io import hopen
from fex.nn import ALBert
from fex.nn.backbone.swin_transformer import SwinTransformer

from fex.matx.text_ops import MultiDomainConcatBuilder, BertInputsBuilder, BertTokenizer


from fuxi.config import cfg
from fuxi.net import NETMAP
from tasks.clip.data.matx_vision_pipeline import MatxVisionGPUPipe

FLAGS = flags.FLAGS


def def_flags():
    flags.DEFINE_string("config_path", '', "config path")
    flags.DEFINE_string("ckpt_path", '', "model")
    flags.DEFINE_string("trace_path", '', "traced")

def stack_and_pad_images(frame0: List[bytes],
                         frame1: List[bytes],
                         frame2: List[bytes],
                         frame3: List[bytes],
                         frame4: List[bytes],
                         frame5: List[bytes],
                         frame6: List[bytes],
                         frame7: List[bytes]) -> Tuple[List[bytes], matx.NDArray]:
    batch_size = len(frame0)
    video_frames = matx.List()
    video_frames.reserve(batch_size * 8)
    frames_mask = matx.List()
    frames_mask.reserve(batch_size * 8)

    all_frames_num: int = 0

    for index in range(batch_size):
        not_empty_frames = matx.List()
        not_empty_frames.reserve(8)
        cur_frames_mask = matx.List()
        cur_frames_mask.reserve(8)

        # 找一个frame，作为后面pad用的
        pad_frame = b''
        for frame in frame0: #TODO: 如果一个batch 内，所有的frame0 都是空的，那就sb了。
            if len(frame) > 0:
                pad_frame = frame
                break
        
        # 按 frame 维度来塞，最后塞出来的是 frame x batch 的大小
        for frames in [frame0, frame1, frame2, frame3, frame4, frame5, frame6, frame7]:
            frame = frames[index]
            if len(frame) > 0:
                not_empty_frames.append(frame)
                cur_frames_mask.append(1)
            else:
                not_empty_frames.append(pad_frame)
                cur_frames_mask.append(0)

        all_frames_num += len(not_empty_frames)
        video_frames.extend(not_empty_frames)
        frames_mask.extend(cur_frames_mask)
    frames_mask_nd = matx.NDArray(frames_mask, [], "int64")
    return video_frames, frames_mask_nd


class TextVisAddNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.albert = ALBert(config.BERT)
        self.projector = torch.nn.Linear(config.BERT.hidden_size, 128)

        self.resnet = SwinTransformer(num_classes=512,
                                      embed_dim=config.NETWORK.embed_dim,
                                      depths=config.NETWORK.depths,
                                      num_heads=config.NETWORK.num_heads)
        self.fc128 = torch.nn.Linear(512, 128, bias=False)

    @torch.no_grad()
    def trace(self,
              input_ids: torch.Tensor,
              input_segment_ids: torch.Tensor,
              input_mask: torch.Tensor,
              frames: torch.Tensor,
              frames_mask: torch.Tensor,
              weight: torch.Tensor,
              ) -> List[torch.Tensor]:
        """
        text: [bsz, seq_len]
        frames: [bsz*, c, h, w]
        """
        bsz = input_ids.shape[0]
        vis_out = self.resnet(frames) # [bsz * frame, dim]
        vis_out = self.fc128(vis_out)
        vis_out = vis_out.reshape(bsz, 8, 128)
        frames_mask = frames_mask.reshape(bsz, 8)
        valid_frame_num = frames_mask.sum(-1, keepdim=True) + 0.001 # 加一个很小的值防止全0
        vis_out = torch.sum(vis_out * frames_mask.unsqueeze(-1), dim=1) / valid_frame_num
        text_out = self.albert(input_ids=input_ids, input_mask=input_mask, input_segment_ids=input_segment_ids)['pooled_output']
        text_out = self.projector(text_out)

        vis_out = torch.nn.functional.normalize(vis_out, dim=1)
        text_out = torch.nn.functional.normalize(text_out, dim=1)
        emb = vis_out * weight.unsqueeze(-1) + text_out * (1 - weight).unsqueeze(-1)
        return emb

class TextAddFramesTraceProcess:
    """
    """

    def __init__(self):
        self.matx_pipe = MatxVisionGPUPipe(image_dsr_height=224,
                                            image_dsr_width=224,
                                            resize_shorter=224,
                                            mean=[0.485 * 255, 0.456 *
                                                  255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 *
                                                 255, 0.225 * 255],
                                            scale=[0.4, 1.0],
                                            ratio=[0.8, 1.25],
                                            thread_num=6,
                                            device_id=int(os.environ.get(
                                                'LOCAL_RANK') or 0),
                                            mode="val",
                                            use_pad=False,
                                            is_trace=True)

        vocab_file = '/mnt/nlp-lq/huangwenguan/zh_old_cut_145607.vocab'
        max_seq_len = 64
        word_piece_tokenizer = matx_text.WordPieceTokenizerOp(location=vocab_file,
                                                              do_wordpiece=True,
                                                              do_lower_case=False)
        self.tokenizer = matx.script(BertTokenizer)(tokenizer=word_piece_tokenizer,
                                                    do_lower_case=False,
                                                    tokenize_emoji=False)


        self.vocab = matx.script(Vocabulary)(vocab_file)
        self.cutter = matx_text.LibcutOp(location=libcut_data_path, cut_type="CRF_LARGE")
        self.multi_domain_concat_builder = matx.script(
            MultiDomainConcatBuilder)(max_seq_len=max_seq_len)

        # 将 batch_inputs_tokens转为input_ids_tensor, segment_ids_tensor和mask_ids_tensor
        self.build_input_builder = matx.script(BertInputsBuilder)(
            max_seq_len=max_seq_len, vocab_file=vocab_file)

        self.preprocess_text = matx.script(PreProcessText)(vocab=self.vocab,
                                                           tokenizer=self.tokenizer,
                                                           cutter=self.cutter,
                                                           concater=self.multi_domain_concat_builder,
                                                           converter=self.build_input_builder
                                                           )


        self.stack_and_pad_images = matx.script(stack_and_pad_images)
        self.model = self.init_model()

    def init_model(self):
        cfg.update_cfg(FLAGS.config_path)
        model = TextVisAddNet(config=cfg)
        load_from_pretrain(model, pretrain_paths=FLAGS.ckpt_path)
        # 2. 将model转为fp16和ft
        convert_ft_and_fp16 = ModelConvertToFTAndFP16(use_fp16=True, use_ft=True)
        fp16_model = convert_ft_and_fp16.convert_model(model)
        fp16_model = matx_pytorch.InferenceOp(model=fp16_model, device=0)
        return fp16_model

    def process(self,
                text: List[bytes],
                frame0: List[bytes],
                frame1: List[bytes],
                frame2: List[bytes],
                frame3: List[bytes],
                frame4: List[bytes],
                frame5: List[bytes],
                frame6: List[bytes],
                frame7: List[bytes],
                weight: matx.NDArray):

        # images
        video_frames, frames_mask = self.stack_and_pad_images(
            frame0, frame1, frame2, frame3, frame4, frame5, frame6, frame7)
        video_frames = self.matx_pipe(video_frames)

        # text
        input_ids, input_segment_ids, input_masks = self.preprocess_text(text)

        emb = self.model(input_ids,
                         input_segment_ids,
                         input_masks,
                         video_frames,
                         frames_mask,
                         weight)
        return emb

    def mock_data(self, batch_size=8):

        import requests
        url = "http://p-risk.byted.org/img/labis/08563748c19a334529ed6f5f00ca9721~320x320.jpeg"
        image = requests.get(url).content

        # im = Image.open('/mnt/nlp-lq/weibaoshan/image/test.jpg')
        # image = image_to_byte_array(im)

        images = [image] * batch_size
        res = {f'frame{i}': images for i in range(8)}
        res['text'] = ['车站'.encode()] * batch_size
        res['weight'] = matx.NDArray([1] * batch_size, [], "int64")
        return res

class Vocabulary:
    def __init__(self, vocab_file: str, unk_token: str = '[UNK]') -> None:
        self.vocab: Dict[str, int] = matx.Dict()
        self.unk: str = unk_token

        fr = open(vocab_file)
        idx = 0
        for token in fr:
            token = token.strip()
            self.vocab[token] = idx
            idx += 1
        fr.close()
        self.unk_id: int = self.vocab[unk_token]

    def lookup(self, key: str) -> int:
        if key in self.vocab:
            return self.vocab[key]
        else:
            return self.unk_id

class PreProcessText:

    def __init__(self,
                 vocab: Vocabulary,
                 tokenizer: matx.NativeObject,
                 cutter: matx.NativeObject,
                 concater: matx.NativeObject,
                 converter: matx.NativeObject,
                 max_seq_len: int = 64) -> None:
        self.vocab: Vocabulary = vocab
        self.max_seq_len: int = max_seq_len
        self.word_piece_tokenizer: matx.NativeObject = tokenizer
        self.cutter: matx.NativeObject = cutter
        self.concater: matx.NativeObject = concater
        self.converter: matx.NativeObject = converter

    def __call__(self, text: List[bytes]) -> Tuple[List, List, List]:
        tokens_batch = matx.List()
        tokens_batch.reserve(len(text))
        for t in text:
            cut_t: List[str] = self.cutter(t.decode(), 'default')
            tokens_batch.append(' '.join(cut_t))
        tokens = self.word_piece_tokenizer(tokens_batch)
        
        # multi domain concat
        input_tokens, segment_ids = self.concater(
            [tokens],
            [0],
            [self.max_seq_len])
        # build bert input tensor
        batch_input_ids_tensor, batch_input_segment_tensor, batch_input_mask_tensor = self.converter(
            input_tokens,
            segment_ids)

        return batch_input_ids_tensor, batch_input_segment_tensor, batch_input_mask_tensor


def try_some(tracer):
    frame_num = 8

    def b64_decode(string):
        if isinstance(string, str):
            string = string.encode()
        return base64.decodebytes(string)

    def gen_raw(raw):
        frames = [[] for _ in range(frame_num)]
        texts = []
        gids = []
        for d in raw:
            texts.append(d['title'].encode())
            gids.append(d['gid'])
            for i in range(1):
                frames[i].append(b64_decode(d['frames'][i]['b64_content']))
            for i in range(1, frame_num):
                frames[i].append('')
        return frames, texts, gids

    batch_size = 8
    with hopen('hdfs://haruna/home/byte_search_nlp_lq/multimodal/data/video/ctr_pos/part-01012.4mz') as f, \
            open('emb.out', 'w') as fo:
        cur_raw = []
        for l in f:
            jl = json.loads(l)
            if len(cur_raw) < batch_size:
                cur_raw.append(jl)
            else:
                frames, texts, gids = gen_raw(cur_raw)
                batch = {f'frame{i}': frames[i] for i in range(frame_num)}
                batch['text'] = texts
                batch['weight'] = matx.NDArray([1] * batch_size, [], "int64")
                trace_out = tracer.Run(batch)
                print(len(trace_out), trace_out[0].shape())
                del cur_raw
                cur_raw = []
                for gid, out, text in zip(gids, trace_out, texts):
                    print(out.shape())
                    out = out.asnumpy()
                    out = out.tolist()
                    fo.write(json.dumps({
                        'gid': gid,
                        'emb': out,
                        'title': text.decode()
                    }, ensure_ascii=False)+'\n')


def traceit(_):
    # trace
    tracer = TextAddFramesTraceProcess()
    jit_module = matx.pipeline.Trace(tracer.process, 
                                     **tracer.mock_data())
    # 设置output key
    jit_module.SetAttr("output_names", ["embedding"])
    jit_module.Save(FLAGS.trace_path)

    gpu = 0
    tracer_loaded = matx.pipeline.Load(FLAGS.trace_path, gpu)

    # calc diff
    try_some(tracer_loaded)

if __name__ == "__main__":
    def_flags()
    app.run(traceit)
