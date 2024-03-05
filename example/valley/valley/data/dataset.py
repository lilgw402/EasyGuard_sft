from concurrent.futures import BrokenExecutor
import json
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import random
import os
import torch
import json
import transformers
from typing import Dict, Sequence
from dataclasses import dataclass
from valley.util.config import *
from valley.util.data_util import preprocess, preprocess_multimodal
import copy
import random
import numpy as np
from torchvision import transforms
import decord
import traceback
import urllib
from io import BytesIO

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, #数据集文件的路径
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args,
                 inference):
        super(LazySupervisedDataset, self).__init__()
        breakpoint()
        #根据传入的 `data_path`，尝试从文件中加载数据
        list_data_dict = []
        if os.path.isfile(data_path) and data_path[-4:] != 'json':
            list_data_dict = [json.loads(data) for data in open(data_path, 'r').readlines()]
        else:
            list_data_dict = json.load(open(data_path, "r"))
        print(list_data_dict[:2])

        #如果设置了 `video_data_path`，它还尝试加载包含视频相关数据的文件
        if data_args.video_data_path is None: #None
            list_video_data_dict = []
        elif os.path.isfile(data_args.video_data_path):
            list_video_data_dict = json.load(open(data_args.video_data_path, "r")) if data_args.video_data_path else []
        else:
            list_video_data_dict = []
            video_data_path_list = os.listdir(data_args.video_data_path)
            for file_name in tqdm(video_data_path_list):
                data_path = os.path.join(data_args.video_data_path, file_name)
                list_video_data_dict += json.load(open(data_path, "r"))
        list_data_dict = list_video_data_dict + list_data_dict
        #如果不处于推理模式，它会对数据列表进行随机混洗
        if not inference:
            random.shuffle(list_data_dict)
        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.inference = inference
    def __len__(self):
        return len(self.list_data_dict) #返回数据集中样本的总数

    @property
    def lengths(self): #计算并返回数据集中每个样本的长度
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self): #另一个属性，它计算并返回一个长度列表，考虑到多模态样本中每个样本的文本长度和图像的存在（如果有）。如果样本中有图像，则长度为正数；如果没有，则长度为负数。
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    #获取给定索引（i）的输入数据，进行处理，并以准备好的格式返回，以便模型训练或推理使用
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i] #获取索引为 `i` 的数据实例(一个样本，包含id，image列表，conversations，label)。
        print("sources===================",sources)
        try:
            if isinstance(i, int):
                sources = [sources]
            assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
            #对于单个图像：它打开并处理位于指定 `image_file` 路径的图像。图像经过处理，转换为模型可以消费的张量（通常是归一化和调整大小）
            if ('image' in sources[0]) and isinstance(self.list_data_dict[i]['image'], str):       ### for single image
                image_file = self.list_data_dict[i]['image']
                image_folder = self.data_args.image_folder
                if 'train2014' in image_folder:
                        image_file = 'COCO_train2014_'+image_file
                processor = self.data_args.image_processor
                image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')

                #expand2square：一个处理图像的函数，根据模型的要求将图像填充成正方形
                if self.data_args.image_aspect_ratio == 'pad':
                    image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                else:
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                #image shape [3,336,336]
                sources = preprocess_multimodal(
                    copy.deepcopy([e["conversations"] for e in sources]),
                    self.data_args)
                image = image.unsqueeze(0)
            #对于多个图像：循环处理每个图像，然后将其堆叠到一个张量中
            elif ('image' in sources[0]) and isinstance(self.list_data_dict[i]['image'], list):     ### for multi image 
                image_list = []
                for image_file in self.list_data_dict[i]['image'][:self.data_args.max_img_num]:
                    image_folder = self.data_args.image_folder if self.data_args.image_folder else ''
                    processor = self.data_args.image_processor
                    # image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
                    try:
                        if self.inference:
                            image_folder = os.path.join(image_folder, self.list_data_dict[i]['id'].split('_')[1])
                        image = read_and_download_img(image_file, image_folder) #图片数据
                    except:
                        print(f'down img err, url: {image_file}')
                        print(traceback.format_exc())
                        image = Image.new(mode="RGB", size=(224, 224))
                    if self.data_args.image_aspect_ratio == 'pad':
                        image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                        image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                    else:
                        image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                    image_list.append(image)
                image_list =  torch.stack(image_list, dim = 0)
                #调用 `preprocess_multimodal` 函数处理文本数据（对话，conversations的部分），主要是处理图片的标识
                sources = preprocess_multimodal(
                    copy.deepcopy([e["conversations"] for e in sources]), #只传入conversations的部分
                    self.data_args)

                image = image_list #一个batch的图片数据
            #对于视频
            elif 'video' in sources[0]:                                                     ### for video file or folder
                video_file = self.list_data_dict[i]['video']
                processor = self.data_args.image_processor
                if 'source' not in self.list_data_dict[i]:
                    video_file = os.path.join(self.data_args.video_folder, video_file)
                else:
                    video_file_source = self.list_data_dict[i]['source']
                    video_file = os.path.join(self.data_args.video_folder, video_file_source, video_file)
                #如果提供了视频文件，它会使用 `decord` 读取视频帧，提取固定数量的帧，像处理图像一样进行处理，然后堆叠到一个张量中。
                if os.path.isfile(video_file):
                    video_reader = decord.VideoReader(video_file, num_threads=1, ctx= decord.cpu(0))
                    decord.bridge.set_bridge('torch')
                    video_len = len(video_reader)
                    video = video_reader.get_batch(np.linspace(0, video_len - 1, 8).astype(np.int_)).byte()  # 8, height,width,3
                #如果提供了视频文件夹，它会读取文件夹中的图像文件，最多到设定的限制数，并像处理单个图像一样进行处理
                else:
                    if os.path.exists(video_file):
                        video = [os.path.join(video_file, file) for file in os.listdir(video_file)][:self.data_args.max_img_num]
                    else:
                        video = []
                    padded_list = ['/mnt/bn/zhaoziwang/multimodal-pretrain-data/demodata/blackimage/black_image.png']*max(8-len(video),0) # this 
                    video = video + padded_list
                video_pad = []
                #对于视频中的每一帧，都会进行与处理图像相同的处理步骤，并将它们堆叠成一个张量
                for image in video:
                    if isinstance(image, str):
                        imagetoPIL = Image.open(image)
                    else:
                        imagetoPIL = transforms.ToPILImage()(image.permute(2,0,1)).convert('RGB')
                    
                    if self.data_args.image_aspect_ratio == 'pad':
                        imagetoPIL = expand2square(imagetoPIL, tuple(int(x*255) for x in processor.image_mean))
                    # processor.preprocess：图像或视频帧被预处理的步骤。这包括根据图像的均值和标准差进行调整大小、归一化和转换为张量。
                    image = processor.preprocess(imagetoPIL, return_tensors='pt')['pixel_values'][0]
                    video_pad.append(image)
                video = torch.stack(video_pad, dim = 0)
                #调用 `preprocess_multimodal` 函数处理文本数据（对话）
                sources = preprocess_multimodal(
                        copy.deepcopy([e["conversations"] for e in sources]),
                        self.data_args)
                image = video
            else:
                sources = copy.deepcopy([e["conversations"] for e in sources])
            #在推理过程中（`self.inference` 为 `True` 时），某些模态可能不会被处理，或者可能有特殊处理以适应不同的模型评估场景。
            if self.inference and len(sources[0])%2 == 0:
                sources[0] = sources[0][:-1]
            #利用 `preprocess` 函数创建了 `data_dict`，它对对话数据进行标记化处理，并可能根据模态对文本数据应用遮蔽
            # 如果模型是多模态的，但当前数据点不包含图像或视频，则创建一个适当维度的零张量来表示空的视觉输入。
            data_dict = preprocess( 
                sources, #对话文本的列表
                self.tokenizer,
                has_image=('image' in self.list_data_dict[i] or 'video' in self.list_data_dict[i]),
                only_mask_system= self.data_args.only_mask_system,
                inference = self.inference)
            #处理完数据后，它被格式化为一个包含 `input_ids`、`labels` 和 `image` 键的字典，分别对应于输入标记、标签标记和图像数据。如果原始数据中有 `id` 或 `label`，这些也会包含在输出字典中。
            if isinstance(i, int):
                data_dict = dict(input_ids=data_dict["input_ids"][0],
                                labels=data_dict["labels"][0])
            # image exist in the data,data_dict中增加image信息
            if 'image' in self.list_data_dict[i] or 'video' in self.list_data_dict[i]:
                data_dict['image'] = image
            elif self.data_args.is_multimodal:
                # image does not exist in the data, but the model is multimodal
                crop_size = self.data_args.image_processor.crop_size
                data_dict['image'] = torch.zeros(1, 3, crop_size['height'], crop_size['width'])
            if 'label' in self.list_data_dict[i]:
                data_dict['label'] = self.list_data_dict[i]['label']
            if 'id' in self.list_data_dict[i]:
                data_dict['id'] = self.list_data_dict[i]['id']
            # print("data_dict==================",data_dict)
            return data_dict
        except Exception as e:
            traceback.print_exc()
            print(self.list_data_dict[i]['id'])
            print(e)
            return ('fail', sources)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        instances_no_error = []
        for ins in instances:
            if type(ins) != tuple and len(ins["input_ids"]) < self.tokenizer.model_max_length:
                instances_no_error.append(ins)
        instances = instances_no_error
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        
        if 'gt_label' in instances[0]:
            gt_label = [instance['gt_label'] for instance in instances]
            batch['gt_label'] = gt_label
        return batch

#创建数据集和数据整合器
def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args, inference = False) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args,
                                inference = inference)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def read_and_download_img(imgurl, image_folder='/mnt/bn/yangmin-priv-fashionmm/Data/wuji/big_model_train_image_data'):
    name = imgurl.split('/')[-1]
    img_path = os.path.join(image_folder, name + f'.png')
    # print(img_path)
    
    if os.path.exists(img_path):
        img_data = Image.open(img_path).convert('RGB')
        # print(img_data)
    else:
        print('image not exist, download it', img_path)
        image_data = urllib.request.urlopen(imgurl, timeout=2).read()
        img_data = Image.open(BytesIO(image_data)).convert('RGB')
        # img_data.save(img_path, format="PNG") #没有权限
    return img_data