"""
trace 一个 clip 模型
"""
from absl import app
from absl import flags

import requests

from fex.model import CLIP
from fex.matx.tokenization import MatxBertTokenizer
from fex.trace.model_convert import ModelConvertToFTAndFP16
from example.trace.bytedvision_pipe_15 import VisionGPUPipe

FLAGS = flags.FLAGS


def def_flags():
    flags.DEFINE_string("config_path", '', "config path")
    flags.DEFINE_string("ckpt_path", '', "model")
    flags.DEFINE_string("trace_path", '', "traced")


class CLIPModelTraceProcess(CLIP):
    """
    定义一个 nn.Moduel， 用来实现一个函数，作为你想trace的模型计算流程
    """

    @torch.no_grad()
    def trace(self,
              input_ids: torch.Tensor,
              input_segment_ids: torch.Tensor,
              input_mask: torch.Tensor,
              image: torch.tensor) -> torch.Tensor:
        """
        输入是文本和图片，输出是分数
        """
        v_out = self.encode_image(image)
        v_emb = torch.nn.functional.normalize(v_emb, dim=-1)
        t_emb = self.encode_text(input_ids=input_ids,
                                 input_segment_ids=input_segment_ids,
                                 input_mask=input_mask)
        t_emb = torch.nn.functional.normalize(t_emb, dim=-1)
        score = (t_em * v_emb).sum(-1)
        return score


class MatxTraceProcess:
    """
    定义一个matx trace 过程，包括预处理和模型计算
    """

    def __init__(self):
        self.tokenizer = matx.script(MatxBertTokenizer)(vocab_path=vocab_path,
                                                        do_cut=True,
                                                        lower_case=True,
                                                        unk_token='[UNK]',
                                                        wordpiece_type='bert')
        self.vision_pipe = VisionGPUPipe(image_dsr_height=224,
                                         image_dsr_width=224,
                                         resize_longer=256,
                                         mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                         std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                                         thread_num=6,
                                         device_id=int(os.environ.get('LOCAL_RANK') or 0)
                                         )
        self.model = self.init_model()

    def init_model(self):
        cfg.update_cfg(FLAGS.config_path)
        model = CLIPModelTraceProcess(config=cfg)
        load_from_pretrain(model, pretrain_paths=FLAGS.ckpt_path)
        # 2. 将model转为fp16和ft
        convert_ft_and_fp16 = ModelConvertToFTAndFP16(use_fp16=True, use_ft=True)
        fp16_model = convert_ft_and_fp16.convert_model(model)
        fp16_model = matx_pytorch.InferenceOp(model=fp16_model, device=0)
        return fp16_model

    def process(self,
                text: List[bytes],
                image: List[bytes]):
        """
        这个是要trace的全过程
        """
        image_nd = self.matx_pipe(image)
        input_ids, input_segment_ids, input_masks = self.tokenizer(text)
        score = self.model(input_ids, input_segment_ids, input_masks, image_nd)
        return score

    def mock_data(self, batch_size=8):
        url = "http://p-risk.byted.org/img/labis/08563748c19a334529ed6f5f00ca9721~320x320.jpeg"
        image = requests.get(url).content
        res = {
            'image': [image] * batch_size,
            'text': ['车站'.encode()] * batch_size
        }
        return res


def traceit():
    matx_trace_process = MatxTraceProcess()
    traced_process = matx.pipeline.Trace(matx_trace_process.process,
                                         **matx_trace_process.mock_data())
    # 设置output key
    traced_process.SetAttr("output_names", ["score"])
    traced_process.Save(FLAGS.trace_path)
    return traced_process


def trysome(traced_process):
    gpu = 0
    tracer_loaded = matx.pipeline.Load(FLAGS.trace_path, gpu)

    with hopen('hdfs://haruna/home/byte_search_nlp_lq/multimodal/data/image/tusou_data_018b_fromq_01/part-00126.4mz') as f:
        for i, l in enumerate(f):
            jl = json.loads(l)
            image = base64.decodebytes(jl['b64_resized_binary'].encode())
            text = jl['title_o']
            score = traced_process.Run({'image': [image], 'text': [text]})
            print(f'image: {jl["image_url"]}')
            print(f'text: {text}')
            print(f'score: {score}')
            if i > 10:
                break


def run(_):
    traced_process = traceit()
    trysome(traced_process)


if __name__ == "__main__":
    def_flags()
    app.run(run)
