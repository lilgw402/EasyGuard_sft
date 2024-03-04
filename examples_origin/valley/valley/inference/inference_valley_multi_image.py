import argparse
import torch
from transformers import LlamaTokenizer
from valley.model.language_model.valley_llama import ValleyProductLlamaForCausalLM
import torch
import os
import re
import json
from valley.utils import disable_torch_init
# from transformers import CLIPImageProcessor
from transformers import ChineseCLIPImageProcessor as CLIPImageProcessor
from PIL import Image
import os
import random
from tqdm import tqdm
from valley.conversation import conv_templates, SeparatorStyle
from valley.util.config import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN, DEFAULT_VIDEO_FRAME_TOKEN, DEFAULT_VI_START_TOKEN, DEFAULT_VI_END_TOKEN
from valley.util.data_util import  KeywordsStoppingCriteria,load_video
from valley.data.dataset import read_and_download_img

def inference(args):
    
    random.seed()

    disable_torch_init()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = os.path.expanduser(args.model_name)
    # tokenizer = LlamaTokenizer.from_pretrained(args.model_name)
    tokenizer = LlamaTokenizer.from_pretrained(os.path.dirname(model_name), use_fast = False)
    
    # load model

    print('load model')
    model = ValleyProductLlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    print('load end')
    
    model.model.multi_image = False
    model.model.multi_image_mode = 'concatenate'
    model = model.to(device)
    image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    vision_tower = model.get_model().vision_tower   
    vision_tower.to(device, dtype=torch.float16)
    vision_config = vision_tower.config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end:
        vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
        vision_config.vi_start_token, vision_config.vi_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_VI_START_TOKEN, DEFAULT_VI_END_TOKEN])
        vision_config.vi_frame_token = tokenizer.convert_tokens_to_ids(DEFAULT_VIDEO_FRAME_TOKEN)

    keywords = ['###', '##']
    
    image_root = args.image_folder

    try:
        inference_data = json.load(open(args.inference_json_path, 'r', encoding="utf-8"))
    except:
        inference_data = []
        with open(args.inference_json_path, 'r', encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                inference_data.append(data)
    data_length = len(inference_data)
    bs = data_length // args.world_size
    rank = args.part_no
    if rank < args.world_size-1:
        local_inference_data = inference_data[rank*bs: (rank+1)*bs]
    else:
        local_inference_data = inference_data[rank*bs:]

    review_file = open(f'{args.out_path}', 'w')
    if not os.path.exists(os.path.dirname(args.out_path)):
        os.makedirs(os.path.dirname(args.out_path))

    ans = []
    max_img_num = 12
    for data in tqdm(local_inference_data):
        # process prompt
        id = data['id']
        gt_label = data['label']
        qs = data['conversations'][0]['value']
        segs = re.split(r'<image[\d]*>', qs)
        qs = '<image>'.join(segs[:max_img_num+1]) + ''.join(segs[max_img_num+1:])
        replace_token = DEFAULT_IMAGE_TOKEN
        if mm_use_im_start_end:
            replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
        qs = qs.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        inputs = tokenizer([prompt])
        input_ids = torch.as_tensor(inputs.input_ids).cuda()
        input_ids[input_ids==tokenizer.encode(DEFAULT_IMAGE_TOKEN)[1]]=-200
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        # load image
        if isinstance(data['image'], str):
            data['image'] = [data['image']]
        if image_root =='none':
            image_files = data['image']
        else:
            image_files = data['image']
        
        images = []
        # print(len(image_files))
        for image_path in image_files:
            try:
                # image = Image.open(image_path).convert('RGB')
                image = read_and_download_img(image_path, os.path.join(image_root, data['id'].split('_')[1]))
            except:
                image = Image.new(mode="RGB", size=(224, 224))
                print('error image', image_path)
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            image_tensor = image.to(device)
            images.append(image_tensor)
        images = torch.stack(images, dim=0)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images.unsqueeze(0).half().cuda(),# 1,8,3,224,224
                do_sample=False,
                temperature=1,
                max_new_tokens=1024,
                stopping_criteria=[stopping_criteria])
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        for i,out in enumerate(outputs):
            while True:
                cur_len = len(out)
                out = out.strip()
                for pattern in ['###', '##', 'Assistant:', 'Response:','LLaVA:', '助理']:
                    if out.startswith(pattern):
                        out = out[len(pattern):].strip()
                if len(out) == cur_len:
                    break
            out = out.replace('\n###','').replace('\n##','').strip()
        # print(out)
        res = [str(id), str(gt_label), out.replace('\n','')]
        review_file.write('    '.join(res) + '\n')
        review_file.flush()

        # cur_ans = {
        #     'image_path': image_files,
        #     'query': data['conversations'][0]['value'],
        #     'answer': out,
        #     'gt': data['conversations'][1]['value']
        # }
        # ans.append(cur_ans)
        # review_file.write(json.dumps(cur_ans, ensure_ascii=False) + '\n')
        # review_file.flush()

    # save inference_data
    # json.dump(ans, open(args.out_path, 'w'), ensure_ascii=False, indent=4)

def gather_result(args):
    num_worker = args.world_size
    with open(args.out_path,'a+') as f:
        for i in range(num_worker):
            with open(args.out_path+".worker_"+str(i),'r') as tf:
                tmp_result = tf.readlines()
            f.writelines(tmp_result)
            os.remove(args.out_path+".worker_"+str(i))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="/mnt/bn/yangmin-priv-fashionmm/Checkpoints/wangzhen/valley/pool_adapter_stage2_three_data_reduced/checkpoint-13300/")
    parser.add_argument("--inference_json_path", type=str, required=False,default="/mnt/bn/yangmin-priv-fashionmm/wangzhen/data/product/product_test_100.json")
    parser.add_argument("--out_path", type=str, required=False,default="/mnt/bn/yangmin-priv-fashionmm/Checkpoints/wangzhen/valley/valley_product_multi_image_inference_test.jsonl")
    parser.add_argument("--image_folder", type=str, required=False,default="none")
    parser.add_argument("--vision-tower", type=str, default="/mnt/bn/yangmin-priv-fashionmm/pretrained/chinese-clip-vit-large-patch14/")
    parser.add_argument("--conv-mode", type=str, default="v0")
    parser.add_argument("--part_no", type=int, default="0")
    parser.add_argument("--world_size", type=int, default="1")
    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids
    inference(args)

    

