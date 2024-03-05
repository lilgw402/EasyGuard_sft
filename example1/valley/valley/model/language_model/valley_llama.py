from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..valley_arch import ValleyVideoMetaModel, ValleyVideoMetaForCausalLM, ValleyProductMetaModel, ValleyProductMetaForCausalLM

from valley.util.data_util import load_video, preprocess_multimodal, KeywordsStoppingCriteria, tokenizer_image_token

from valley import conversation as conversation_lib

from PIL import Image

class ValleyConfig(LlamaConfig):
    model_type = "valley"


class ValleyVideoLlamaModel(ValleyVideoMetaModel, LlamaModel):
    config_class = ValleyConfig

    def __init__(self, config: LlamaConfig):
        super(ValleyVideoLlamaModel, self).__init__(config)


class ValleyVideoLlamaForCausalLM(LlamaForCausalLM, ValleyVideoMetaForCausalLM):
    config_class = ValleyConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = ValleyVideoLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.image_processor = self.model.vision_tower.image_processor

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def build_inputs(self, tokenizer, messages, num_image=1, image_token_len = 224,if_context=False, conv = None):
        tokenizer.padding_side = 'left'
        prompt = ''
        sources = messages.copy()
        for sentence in sources:
            sentence['value'] = sentence['content']
            sentence.pop('content')
        
        messages = preprocess_multimodal([sources],{'is_multimodal': True})[0]
        
        roles = {"user": conv.roles[0], "assistant": conv.roles[1]}

        for i, message in enumerate(messages):
            if message["role"] == 'system':
                conv.system = message["value"]
                messages = messages[1:]
                break

        conv.messages = []
        if conv.sep_style == conversation_lib.SeparatorStyle.PLAIN:
            for j, sentence in enumerate(messages):
                role = roles[sentence["role"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
        else:
            for j, sentence in enumerate(messages):
                role = roles[sentence["role"]]
                conv.append_message(role, sentence["value"])
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        print(prompt)
        input_id = tokenizer_image_token(prompt, tokenizer, return_tensors='pt', image_token_len = image_token_len, num_image = num_image)
        return input_id
    
    def process_response(self,outputs):
        output = []
        for i, out in enumerate(outputs):
            while True:
                cur_len = len(out)
                out = out.strip()
                for pattern in ['###', 'Assistant:', 'Response:', 'Valley:']:
                    if out.startswith(pattern):
                        out = out[len(pattern):].strip()
                if len(out) == cur_len:
                    break
            try:
                index = out.index('###')
            except ValueError:
                out += '###'
                index = out.index("###")
            out = out[:index].strip()
            output.append(out)
        return output

    @torch.no_grad()
    def completion(self, tokenizer, video: str, image: str ,message: list, gen_kwargs:dict, device, frame_mode='fixed',fps=0.5,fixed_frame_number=8, conv_mode = 'v1'):
        if video:
            images = load_video(video, self.image_processer, frame_mode=frame_mode, fps_number= fps, fixed_frame_number= fixed_frame_number)
            images = images.permute(1, 0, 2, 3)
            images = images.unsqueeze(0).half().to(device)
            print(images.shape)
        elif image:
            if isinstance(image, list) and isinstance(image[0], str):
                image = [Image.open(img) for img in image]
            elif isinstance(image, list) and not isinstance(image[0], str):
                image = [img for img in image]
            elif isinstance(image, str):
                image = [Image.open(image)]
            else:
                image = [image]

            images = self.image_processer.preprocess(
                image, return_tensors='pt')['pixel_values'].unsqueeze(0).half().to(device)
            # images = images.permute(1, 0, 2, 3)
            # print(images.shape)
        else:
            images = None

        conv = conversation_lib.conv_templates[conv_mode].copy()
        
        inputs = self.build_inputs(tokenizer, message, images.shape[1] if images is not None else 1, image_token_len = (images.shape[-1]//14)**2, if_context= isinstance(image, list), conv = conv)
        input_ids = inputs.unsqueeze(0).to(device)
        
        stop_str = conv.sep if conv.sep_style != conversation_lib.SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
        output_ids = self.generate(input_ids = input_ids, images = images, stopping_criteria=[stopping_criteria],**gen_kwargs)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        # print(outputs)
        response = self.process_response(outputs)
        return response
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs


class ValleyProductLlamaModel(ValleyProductMetaModel, LlamaModel):
    config_class = ValleyConfig

    def __init__(self, config: LlamaConfig):
        super(ValleyProductLlamaModel, self).__init__(config)

#商品多模态大模型
#类派生自 `LlamaForCausalLM` 和 `ValleyProductMetaForCausalLM`，这意味着它可能集成了 LLM (大型语言模型) 的功能，并添加了与 `ValleyProduct` 相关的特定特性。
class ValleyProductLlamaForCausalLM(LlamaForCausalLM, ValleyProductMetaForCausalLM):
    config_class = ValleyConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = ValleyProductLlamaModel(config) #是模型的主干网络，在sft的时候会被freeze_backbone，在contuin training的时候不会被冻结

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False) #初始化 `lm_head`，用于生成最终预测词汇表上的分布的线性层。

        self.image_processor = self.model.vision_tower.image_processor #图像处理部分

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    #根据给定的文本消息和图像数量构建模型输入
    def build_inputs(self, tokenizer, messages, num_image=1, image_token_len = 224,if_context=False, conv = None):
        tokenizer.padding_side = 'left'
        prompt = ''
        sources = messages.copy()
        for sentence in sources:
            sentence['value'] = sentence['content']
            sentence.pop('content')
        
        #对消息进行预处理，这可能涉及将文本信息与图像信息整合为模型可接收的格式。此函数调用表明输入是多模态的。
        messages = preprocess_multimodal([sources],{'is_multimodal': True})[0]
        #建立一个角色映射字典 `roles`，用于转换消息中提及的角色到对话实例(`conv`)中指定的角色
        roles = {"user": conv.roles[0], "assistant": conv.roles[1]}

        for i, message in enumerate(messages):
            if message["role"] == 'system':
                conv.system = message["value"]
                messages = messages[1:]
                break

        conv.messages = []
        if conv.sep_style == conversation_lib.SeparatorStyle.PLAIN:
            for j, sentence in enumerate(messages):
                role = roles[sentence["role"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
        else:
            for j, sentence in enumerate(messages):
                role = roles[sentence["role"]]
                conv.append_message(role, sentence["value"])
        conv.append_message(conv.roles[1], None) #在消息列表的末尾添加一个角色为 `assistant` 的空消息
        prompt = conv.get_prompt() #生成一个输入提示 `prompt`，使用 `conv.get_prompt()` 方法从对话实例中获取。
        print(prompt)
        #将生成的 `prompt` 与图像token整合，并返回tensor格式(`'pt'`)的模型输入ID列表
        input_id = tokenizer_image_token(prompt, tokenizer, return_tensors='pt', image_token_len = image_token_len, num_image = num_image)
        return input_id
    
    #处理输出结果
    def process_response(self,outputs): # outputs：预期为模型生成的一系列文本输出的列表
        output = [] #将用来存放处理过的输出的空列表
        for i, out in enumerate(outputs):
            while True:
                cur_len = len(out)
                out = out.strip() #移除字符串两端的空白字符
                #迭代移除特定的标记或关键词（例如 '###', 'Assistant:', 'Response:', 'Valley:'），这些可能是模型生成文本中的特殊分隔符或提示词。
                for pattern in ['###', 'Assistant:', 'Response:', 'Valley:']:
                    if out.startswith(pattern):
                        out = out[len(pattern):].strip()
                if len(out) == cur_len:
                    break
            #尝试找到字符串 '###'，这可能是输出文本中的结束标记。如果找不到，会在输出的末尾添加一个 '###'，然后截取字符串直到这个标记，确保输出不包括任何不必要的内容。
            try:
                index = out.index('###')
            except ValueError:
                out += '###'
                index = out.index("###")
            out = out[:index].strip()
            output.append(out) #处理和截取后的输出被添加到 `output` 列表中
        return output

    @torch.no_grad()
    def completion(self, tokenizer, video: str, image: str ,message: list, gen_kwargs:dict, device, frame_mode='fixed',fps=0.5,fixed_frame_number=8, conv_mode = 'v1'):
        #根据 `video` 或 `image` 参数的情况处理输入的图像数据。使用 `load_video` 或 `self.image_processer.preprocess` 来处理视频或图像，并在处理后将数据放到 GPU 上（如果有
        if video:
            images = load_video(video, self.image_processer, frame_mode=frame_mode, fps_number= fps, fixed_frame_number= fixed_frame_number)
            images = images.permute(1, 0, 2, 3)
            images = images.unsqueeze(0).half().to(device)
            print(images.shape)
        elif image:
            if isinstance(image, list) and isinstance(image[0], str):
                image = [Image.open(img) for img in image]
            elif isinstance(image, list) and not isinstance(image[0], str):
                image = [img for img in image]
            elif isinstance(image, str):
                image = [Image.open(image)]
            else:
                image = [image]

            images = self.image_processer.preprocess(
                image, return_tensors='pt')['pixel_values'].unsqueeze(0).half().to(device)
            # images = images.permute(1, 0, 2, 3)
            # print(images.shape)
        else:
            images = None

        #获得一个对话模板 `conv`，这是一个预定义的结构，用于构建包含文本和图像的对话提示
        conv = conversation_lib.conv_templates[conv_mode].copy()
        #使用 `build_inputs` 方法构建模型的输入，并将输入数据发送到 GPU
        inputs = self.build_inputs(tokenizer, message, images.shape[1] if images is not None else 1, image_token_len = (images.shape[-1]//14)**2, if_context= isinstance(image, list), conv = conv)
        input_ids = inputs.unsqueeze(0).to(device)
        
        stop_str = conv.sep if conv.sep_style != conversation_lib.SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
        #调用 `generate` 方法进行文本生成，传递输入 IDs、图像和停止标准等参数
        output_ids = self.generate(input_ids = input_ids, images = images, stopping_criteria=[stopping_criteria],**gen_kwargs)

        #对生成的输出进行分析，检查输入和生成的输出之间是否有差异
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        #使用 `tokenizer.batch_decode` 对生成的文本张量进行解码，获取可读的文本输出。
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        # print(outputs)
        response = self.process_response(outputs)
        return response
    
    def forward(
        self,
        input_ids: torch.LongTensor = None, #输入的token ids
        attention_mask: Optional[torch.Tensor] = None, #用于标识序列中哪些部分不应参与注意力计算的掩码
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None, #决定是否使用缓存机制来存储计算的键值对，用于加速生成
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None, #像数据，用于多模态输入。
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states) #通过线性层计算预测的词表分布（logits）

        loss = None
        #如果提供了真实标签，计算损失。损失计算通常涉及将预测向前移动一位，即让第 n-1 个token去预测第 n 个token
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        #根据`return_dict`返回一个封装的输出字典或一个元组，包含了损失和其他可能的输出
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    #用于文本生成的，当模型在解码或生成文本时被调用
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        #如果存在`past_key_values`,表明是解码过程的后续步骤，只关注输入ID序列的最后一个token
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs


AutoConfig.register("valley", ValleyConfig)
AutoModelForCausalLM.register(ValleyConfig, ValleyVideoLlamaForCausalLM)
AutoModelForCausalLM.register(ValleyConfig, ValleyProductLlamaForCausalLM)
