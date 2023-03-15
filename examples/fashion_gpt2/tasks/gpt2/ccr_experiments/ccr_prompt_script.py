# -*- coding: utf-8 -*-
import json
import random
from typing import Dict, List, Tuple

import pandas as pd
import regex


def load_mlc_corpus(input_file_path: str) -> List[Tuple]:
    ret = []
    with open(file=input_file_path, mode='r', encoding='utf-8') as fr:
        for line in fr:
            arr_line = line.strip().split("\t")
            label_path, text = arr_line[0].split(","), arr_line[1]
            ret.append((label_path, text))
    print(f"SUCCESSFULLY LOADED {len(ret)} DATA POINTS FROM: {input_file_path}")
    return ret


def load_label_definition(input_file_path: str) -> Dict[str, str]:
    ret = {}
    with open(file=input_file_path, mode='r', encoding='utf-8') as fr:
        for line in fr:
            arr_line = line.strip().split("\t")
            label, definition = arr_line[0], arr_line[1]
            if label in ret:
                raise ValueError(f"Duplicate label name: {label}")
            ret[label] = definition
    print(f"SUCCESSFULLY LOADED {len(ret)} DATA POINTS FROM: {input_file_path}")
    return ret


def load_label_map(input_file_path: str) -> Dict[str, str]:
    ret = {}
    with open(file=input_file_path, mode='r', encoding='utf-8') as fr:
        for line in fr:
            arr_line = line.strip().split("\t")
            label, definition, label_code = arr_line[0], arr_line[1], arr_line[2]
            if label in ret:
                raise ValueError(f"Duplicate label name: {label}")
            ret[label] = label_code
    print(f"SUCCESSFULLY LOADED {len(ret)} DATA POINTS FROM: {input_file_path}")
    return ret


# def generate_mlc_prompt(input_file_path: List) -> List[str]:
#     corpus = load_mlc_corpus(input_file_path=input_file_path)
#     ret = []
#     for data_point in corpus:
#         prompted_data = f"以下这条电商平台用户评论: {data_point[0]}, 命中了多个CCR负向标签，分别是：{data_point[1]}"
#         ret.append(json.dumps({"page_info": {'core_content': prompted_data}}, ensure_ascii=False))
#     return ret

def generate_valid_prompt_data(input_file_path: str,
                               label_definition_path: str,
                               valid_out_file_path: str):
    label_definition_dict = load_label_definition(label_definition_path)
    with open(input_file_path, mode='r', encoding='utf-8') as fr, \
            open(valid_out_file_path, mode='w', encoding='utf-8') as fw:
        for i, line in enumerate(fr):
            arr_line = line.strip().split("\t")
            labels, text = arr_line[0].split(","), arr_line[1]
            for k, v in label_definition_dict.items():
                label_definition = label_definition_dict[k]
                prompted_question = f"根据以下电商CCR标签定义:\"{label_definition}\"; 这句用户评价:\"{text}\"" \
                                    f"是否满足该标签定义(A:满足、B:不满足)-->"
                answer = "A" if k in labels else "B"
                data_obj = {
                    "page_info"       : {
                        "query" : prompted_question,
                        "answer": answer
                    },
                    "label_definition": label_definition,
                    "label"           : k
                }
                fw.write(json.dumps(data_obj, ensure_ascii=False) + "\n")


def generate_train_prompt_data(input_file_path: str, label_definition_path: str,
                               parquet_out_file_path: str, txt_out_file_path: str) -> None:
    label_definition_dict = load_label_definition(label_definition_path)
    label_definition_list = list(label_definition_dict.values())

    corpus = load_mlc_corpus(input_file_path=input_file_path)
    data_frame = []
    for data in corpus:
        labels, text = data[0], data[1]
        for label in labels:
            if label != "非负向:非负向:非负向":
                label_definition = label_definition_dict[label]
                prompted_question = f"根据以下电商CCR标签定义:\"{label_definition}\"; 这句用户评价:\"{text}\"" \
                                    f"是否满足该标签定义(A:满足、B:不满足)-->"
                answer = "A"
                data_frame.append((prompted_question, answer))

                random_label_definitions = random.sample(label_definition_list, 3)
                for random_label_definition in random_label_definitions:
                    if random_label_definition != label_definition:
                        prompted_question = f"根据以下电商CCR标签定义:\"{random_label_definition}\"; 这句用户评价:\"{text}\"" \
                                            f"是否满足该标签定义(A:满足、B:不满足)-->"
                        answer = "B"
                        data_frame.append((prompted_question, answer))
            else:
                random_label_definition = random.sample(label_definition_list, 1)[0]
                prompted_question = f"根据以下电商CCR标签定义:\"{random_label_definition}\"; 这句用户评价:\"{text}\"" \
                                    f"是否满足该标签定义(A:满足、B:不满足)-->"
                answer = "B"
                data_frame.append((prompted_question, answer))
    print(f"TOTAL DATA IN INITIAL CORPUS: {len(corpus)}, TOTAL DATA IN PROMPTED CORPUS: {len(data_frame)}")
    random.shuffle(data_frame)

    with open(txt_out_file_path, mode='w', encoding='utf-8') as fw:
        for data in data_frame:
            fw.write("\t".join(data) + "\n")

    df = pd.DataFrame(data_frame, columns=['question', 'answer'])
    df.to_parquet(parquet_out_file_path)


def generate_prompt_multi_choice_data(input_train_file_path: str,
                                      input_valid_file_path: str,
                                      train_out_file_path: str,
                                      valid_out_file_path: str,
                                      label_map_file_path: str) -> None:
    train_corpus = load_mlc_corpus(input_file_path=input_train_file_path)
    valid_corpus = load_mlc_corpus(input_file_path=input_valid_file_path)
    label2code = load_label_map(input_file_path=label_map_file_path)
    label_codes = "、".join(label2code.values())
    data_frame = []
    max_len = 0
    for data in train_corpus:
        labels, text = data[0], data[1]
        # label_text = ",".join([_.split(":")[-1] for _ in labels])
        label_text = ",".join([label2code[_] for _ in labels])
        prompted_question = f"以下文本\"{text}\"的标签为(选项为:{label_codes})-->"
        answer = f"{label_text}eos"
        all_text = prompted_question + answer
        max_len = max(max_len, len(all_text))
        data_frame.append((prompted_question, answer))
    print(f"maximum data length: {max_len}")

    df = pd.DataFrame(data_frame, columns=['question', 'answer'])
    df.to_parquet(train_out_file_path)

    with open(file=valid_out_file_path, mode='w', encoding='utf-8') as fw:
        for data in valid_corpus:
            labels, text = data[0], data[1]
            label_text = ",".join([label2code[_] for _ in labels])
            prompted_question = f"以下文本\"{text}\"的标签为(选项为:{label_codes})-->"
            answer = f"{label_text}eos"
            data_obj = {
                "page_info": {
                    "query" : prompted_question,
                    "answer": answer
                }
            }
            fw.write(json.dumps(data_obj, ensure_ascii=False) + "\n")


def generate_prompt_mlc_data(input_train_file_path: str,
                             input_valid_file_path: str,
                             train_out_file_path: str,
                             valid_out_file_path: str) -> None:
    train_corpus = load_mlc_corpus(input_file_path=input_train_file_path)
    valid_corpus = load_mlc_corpus(input_file_path=input_valid_file_path)
    data_frame = []
    for data in train_corpus:
        labels, text = data[0], data[1]
        label_text = ",".join([_.split(":")[-1] for _ in labels])
        prompted_question = f"以下用户评价:{text}的标签为:"
        answer = f"{label_text}eos"
        data_frame.append((prompted_question, answer))

    df = pd.DataFrame(data_frame, columns=['question', 'answer'])
    df.to_parquet(train_out_file_path)

    with open(file=valid_out_file_path, mode='w', encoding='utf-8') as fw:
        for data in valid_corpus:
            labels, text = data[0], data[1]
            label_text = ",".join([_.split(":")[-1] for _ in labels])
            prompted_question = f"以下用户评价:{text}的标签为:"
            answer = f"{label_text}eos"
            data_obj = {
                "page_info": {
                    "query" : prompted_question,
                    "answer": answer
                }
            }
            fw.write(json.dumps(data_obj, ensure_ascii=False) + "\n")


def parse_result_from_mlc(input_file_path: str) -> None:
    match = 0
    total = 0
    with open(file=input_file_path, mode='r', encoding='utf-8') as fr:
        for line in fr:
            arr_line = line.strip().split("\t")
            if len(arr_line) > 2:
                text, label, prediction = arr_line[0], arr_line[1], arr_line[2]
            else:
                text, label, prediction = arr_line[0], arr_line[1], "非负向:非负向:非负向"
            label_set = set(label.split(","))
            prediction_set = set(prediction.split(","))
            if label_set == prediction_set:
                match += 1
            total += 1
        print(f"ACC: {match / total}")


def parse_result_mlc_prompt(input_file_path: str) -> None:
    match = 0
    total = 0
    with open(file=input_file_path, mode='r', encoding='utf-8') as fr:
        for line in fr:
            arr_line = line.strip().split("\t")
            text, label, completion = arr_line[0], arr_line[1], arr_line[2]
            label = label.replace("eos", "")
            try:
                prediction = regex.search("的标签为:(.*?)eos", completion, regex.IGNORECASE).group(1)
            except Exception as e:
                total += 1
                continue
            label_set = set(label.split(","))
            prediction_set = set(prediction.split(","))
            if label_set == prediction_set:
                match += 1
            total += 1
        print(f"ACC: {match / total}")


def parse_result_for_prompt(input_file_path: str) -> None:
    match = 0
    total = 0
    text_and_result = {}
    with open(file=input_file_path, mode='r', encoding='utf-8') as fr:
        for line in fr:
            arr_line = line.strip().split("\t")
            prompt_text, label_name, label, generation_text = arr_line[0], arr_line[1], arr_line[2], arr_line[3]
            text = regex.search("这句用户评价:\"(.*)\"", prompt_text, regex.IGNORECASE).group(1)
            if text != "掉毛 第一次用掉了好多毛毛在脸上 差了 搞得皮肤都要过敏了":
                continue
            if text not in text_and_result:
                text_and_result[text] = {
                    "labels"     : set(),
                    "predictions": set()
                }
            if label.lower() == 'a':
                text_and_result[text]['labels'].add(label_name)
            prediction = generation_text.split("-->")[1]
            if prediction.lower() == 'a':
                text_and_result[text]['predictions'].add(label_name)
    for text, info in text_and_result.items():
        if info['labels'] == info['predictions']:
            match += 1
        total += 1
    print(f"ACC: {match / total}")


if __name__ == "__main__":
    data_base_dir = '/Users/bytedance/PycharmProjects/modelzoo/wan_models/classification_models/experiments/ccr_industry/item_model/corpus'
    # generate_prompt_data(input_train_file_path=os.path.join(data_base_dir, 'train.txt'),
    #                      input_valid_file_path=os.path.join(data_base_dir, 'valid.txt'),
    #                      train_out_file_path='ccr_train.parquet',
    #                      valid_out_file_path='ccr_mlc_valid.jsonl')
    # generate_prompt_multi_choice_data(input_train_file_path=os.path.join(data_base_dir, 'train.txt'),
    #                                   input_valid_file_path=os.path.join(data_base_dir, 'valid.txt'),
    #                                   train_out_file_path='ccr_multi_choice_train.parquet',
    #                                   valid_out_file_path='ccr_multi_choice_valid.jsonl',
    #                                   label_map_file_path='label_map.txt')
    # generate_train_prompt_data(input_file_path=os.path.join(data_base_dir, 'train.txt'),
    #                            label_definition_path='label_definition.txt',
    #                            parquet_out_file_path='ccr_train.parquet',
    #                            txt_out_file_path='ccr_train.txt')

    # generate_valid_prompt_data(input_file_path=os.path.join(data_base_dir, 'toy_valid.txt'),
    #                            label_definition_path='label_definition.txt',
    #                            valid_out_file_path='ccr_mlc_valid.jsonl')
    # parse_result_from_mlc(input_file_path='eval_detail.txt')
    parse_result_mlc_prompt(input_file_path='inference_result_2023031513.txt')
