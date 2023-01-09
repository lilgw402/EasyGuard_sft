from transformers import BertModel, BertTokenizer, BertConfig
import torch
import numpy as np
import argparse
import os
from shutil import copy2

from titan.utils.hdfs import (
    hdfs_put,
    has_hdfs_path_prefix
)
from titan.utils.misc import has_http_path_prefix


def save_model(tmp_path, final_path):
    if has_hdfs_path_prefix(final_path):
        hdfs_put(tmp_path, final_path, overwrite=True)
    elif has_http_path_prefix(final_path):
        raise TypeError(f'http path:{final_path} is not supported.')
    else:  # local path
        if not os.path.exists(os.path.dirname(final_path)):
            os.makedirs(os.path.dirname(final_path))
        copy2(tmp_path, final_path)


def bert2torchscript(tokens_tensor, segments_tensors, verify, mname, output_dir):
    # Initializing the model with the torchscript flag
    # Flag set to True even though it is not necessary as this model does not have an LM Head.
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, torchscript=True)

    # Instantiating the model
    model = BertModel(config)

    # The model needs to be in evaluation mode
    model.eval()

    # If you are instantiating the model with `from_pretrained` you can also easily set the TorchScript flag
    model = BertModel.from_pretrained(mname, torchscript=True)

    
    # Creating the trace
    traced_model = torch.jit.trace(model, [tokens_tensor,segments_tensors])

    if not os.path.exists('./tmp'):
        os.mkdir('./tmp')
    tmp_output_path = './tmp/traced_bert.pt'
    traced_model.save(tmp_output_path)

    if verify:
        # check by TS
        loaded = torch.jit.load(tmp_output_path)
        tokenizer = BertTokenizer.from_pretrained(mname)
        inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        print(inputs)
        # check the numerical value
        # get pytorch output
        with torch.no_grad():
            pytorch_result = model(**inputs)
            pytorch_result = pytorch_result[0].cpu().detach().numpy()
        print(pytorch_result.shape)
        token_TS = inputs['input_ids']
        segment_TS = inputs['attention_mask']
        # get TS output
        out_TS = loaded(token_TS,segment_TS)
        out_TS = out_TS[0].cpu().detach().numpy()
    
        # compare results
        np.testing.assert_allclose(
            pytorch_result.astype(np.float32),
            out_TS.astype(np.float32),
            rtol=1e-5,
            atol=1e-5,
            err_msg='The outputs are different between Pytorch and TorchScript')
        print('The outputs are same between Pytorch and TorchScript')

    # save output to final output path
    output_path = os.path.join(output_dir, 'traced_bert.pt')
    save_model(tmp_output_path, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BERT Model Conversion (Torch -> TorchScript)')
    parser.add_argument("-mname", "--model_name", default="bert-base-chinese")
    parser.add_argument('--model_output_dir', type=str, required=True,
                        help='model output directory on hdfs or local')
    parser.add_argument("-v","--verify",default=False)
    args = parser.parse_args()

    enc = BertTokenizer.from_pretrained(args.model_name)

    # Tokenizing input text
    text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
    tokenized_text = enc.tokenize(text)
  
    # Masking one of the input tokens
    masked_index = 8
    tokenized_text[masked_index] = '[MASK]'
    indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)
    
    segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    # Creating a dummy input
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
 
    bert2torchscript(
        tokens_tensor,
        segments_tensors,
        args.verify,
        args.model_name,
        args.model_output_dir)
