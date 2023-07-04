# -*- coding: utf-8 -*-
import json
import random
from typing import Dict, List, Tuple

import pandas as pd
import regex


def load_mlc_corpus(input_file_path: str) -> List[Tuple]:
    ret = []
    with open(file=input_file_path, mode="r", encoding="utf-8") as fr:
        for line in fr:
            arr_line = line.strip().split("\t")
            label_path, text = arr_line[0].split(","), arr_line[1]
            ret.append((text, label_path))
    print(f"SUCCESSFULLY LOADED {len(ret)} DATA POINTS FROM: {input_file_path}")
    return ret


def load_ner_corpus(input_file_path: str) -> List[Tuple]:
    ret = []
    with open(file=input_file_path, mode="r", encoding="utf-8") as fr:
        json_list = json.load(fr)
        for json_obj in json_list:
            text = json_obj["user_text"][0].replace("\n", ",")
            label = ",".join(_.replace("\n", ",") for _ in json_obj["tag_text"])
            if len(text) > 1024:
                continue
            if text.strip() == "" or label.strip() == "":
                print(json_obj)
            ret.append((text, label))
        print(f"SUCCESSFULLY LOADED {len(ret)} DATA POINTS FROM: {input_file_path}")
        return ret


def load_label_definition(input_file_path: str) -> Dict[str, str]:
    ret = {}
    with open(file=input_file_path, mode="r", encoding="utf-8") as fr:
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
    with open(file=input_file_path, mode="r", encoding="utf-8") as fr:
        for i, line in enumerate(fr):
            arr_line = line.strip().split("\t")
            label, definition, label_code = (
                arr_line[0],
                arr_line[1],
                str(i).zfill(3),
            )
            if label in ret:
                raise ValueError(f"Duplicate label name: {label}")
            ret[label] = label_code
    print(f"SUCCESSFULLY LOADED {len(ret)} DATA POINTS FROM: {input_file_path}")
    return ret


def generate_prompt_ner_data(
    input_train_file_path: str,
    input_valid_file_path: str,
    train_out_file_path: str,
    valid_out_file_path: str,
) -> None:
    train_corpus = load_ner_corpus(input_file_path=input_train_file_path)
    valid_corpus = load_ner_corpus(input_file_path=input_valid_file_path)

    def generate_prompt_input(question, answer):
        prompted_question = f'【关键词抽取任务】以下用户问题:"{question}"包含的关键词有:'
        prompted_answer = f"{answer}eos"
        return prompted_question, prompted_answer

    with open(file=train_out_file_path, mode="w", encoding="utf-8") as fw:
        for data in train_corpus:
            text, label = data[0], data[1]
            prompted_question, prompted_answer = generate_prompt_input(question=text, answer=label)
            fw.write("\t".join([prompted_question, prompted_answer]) + "\n")

    with open(file=valid_out_file_path, mode="w", encoding="utf-8") as fw:
        for data in valid_corpus:
            text, label = data[0], data[1]
            prompted_question, prompted_answer = generate_prompt_input(question=text, answer=label)
            data_obj = {
                "page_info": {
                    "query": prompted_question,
                    "answer": prompted_answer,
                }
            }
            fw.write(json.dumps(data_obj, ensure_ascii=False) + "\n")


def generate_prompt_mlc_data(
    input_train_file_path: str,
    input_valid_file_path: str,
    train_out_file_path: str,
    valid_out_file_path: str,
) -> None:
    train_corpus = load_mlc_corpus(input_file_path=input_train_file_path)
    valid_corpus = load_mlc_corpus(input_file_path=input_valid_file_path)

    def generate_prompt_input(question, answer):
        leaf_label = ",".join([_.split(":")[-1] for _ in answer])
        prompted_question = f'【多标签分类任务】以下用户问题:"{question}"的标签为:'
        prompted_answer = f"{leaf_label}eos"
        return prompted_question, prompted_answer

    with open(file=train_out_file_path, mode="w", encoding="utf-8") as fw:
        for data in train_corpus:
            text, labels = data[0], data[1]
            prompted_question, prompted_answer = generate_prompt_input(question=text, answer=labels)
            fw.write("\t".join([prompted_question, prompted_answer]) + "\n")

    with open(file=valid_out_file_path, mode="w", encoding="utf-8") as fw:
        for data in valid_corpus:
            text, labels = data[0], data[1]
            prompted_question, prompted_answer = generate_prompt_input(question=text, answer=labels)
            data_obj = {
                "page_info": {
                    "query": prompted_question,
                    "answer": prompted_answer,
                }
            }
            fw.write(json.dumps(data_obj, ensure_ascii=False) + "\n")


def convert_txt_to_parquet(input_file_path: str, output_file_path: str) -> None:
    data_frame = []
    max_len = 0
    with open(file=input_file_path, mode="r", encoding="utf-8") as fr:
        for line in fr:
            arr_line = line.strip().split("\t")
            question, answer = arr_line[0], arr_line[1]
            data_frame.append((question, answer))
            max_len = max(max_len, len(question + answer))
    df = pd.DataFrame(data_frame, columns=["question", "answer"])
    df.to_parquet(output_file_path)
    print(f"Data frame size: {len(data_frame)}, maximum data length: {max_len}")


def generate_prompt_multi_choice_data(
    input_train_file_path: str,
    input_valid_file_path: str,
    train_out_file_path: str,
    valid_out_file_path: str,
    label_map_file_path: str,
) -> None:
    train_corpus = load_mlc_corpus(input_file_path=input_train_file_path)
    valid_corpus = load_mlc_corpus(input_file_path=input_valid_file_path)
    label2code = load_label_map(input_file_path=label_map_file_path)
    label_codes = "、".join(label2code.values())
    data_frame = []
    max_len = 0
    for data in train_corpus:
        text, labels = data[0], data[1]
        label_text = ",".join([label2code[_] for _ in labels])
        prompted_question = f'以下文本"{text}"的标签为(选项为:{label_codes})-->'
        answer = f"{label_text}eos"
        all_text = prompted_question + answer
        max_len = max(max_len, len(all_text))
        data_frame.append((prompted_question, answer))
    print(f"maximum data length: {max_len}")

    df = pd.DataFrame(data_frame, columns=["question", "answer"])
    df.to_parquet(train_out_file_path)

    with open(file=valid_out_file_path, mode="w", encoding="utf-8") as fw:
        for data in valid_corpus:
            text, labels = data[0], data[1]
            label_text = ",".join([label2code[_] for _ in labels])
            prompted_question = f'以下文本"{text}"的标签为(选项为:{label_codes})-->'
            answer = f"{label_text}eos"
            data_obj = {"page_info": {"query": prompted_question, "answer": answer}}
            fw.write(json.dumps(data_obj, ensure_ascii=False) + "\n")


def parse_result_from_mlc(input_file_path: str) -> None:
    match = 0
    total = 0
    with open(file=input_file_path, mode="r", encoding="utf-8") as fr:
        for line in fr:
            arr_line = line.strip().split("\t")
            if len(arr_line) > 2:
                text, label, prediction = arr_line[0], arr_line[1], arr_line[2]
            else:
                text, label, prediction = (
                    arr_line[0],
                    arr_line[1],
                    "非负向:非负向:非负向",
                )
            label_set = set(label.split(","))
            prediction_set = set(prediction.split(","))
            if label_set == prediction_set:
                match += 1
            total += 1
        print(f"ACC: {match / total}")


def parse_result_mlc_prompt(input_file_path: str) -> None:
    match, total = 0, 0
    with open(file=input_file_path, mode="r", encoding="utf-8") as fr:
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

            if len(prediction_set) == 1 and "非非负向" in prediction_set:
                prediction_set = {"非负向"}

            if label_set == prediction_set:
                match += 1
            total += 1
        print(f"ACC: {match / total}")


def parse_result_ner_prompt(input_file_path: str) -> None:
    match, subset, total = 0, 0, 0
    with open(file=input_file_path, mode="r", encoding="utf-8") as fr:
        for line in fr:
            arr_line = line.strip().split("\t")
            if "【关键词抽取任务】" not in arr_line[0]:
                continue
            text, label, completion = arr_line[0], arr_line[1], arr_line[2]
            label = label.replace("eos", "")
            try:
                prediction = regex.search("的关键词有:(.*?)eos", completion, regex.IGNORECASE).group(1)
            except Exception as e:
                total += 1
                continue
            label_set = set(label.split(","))
            prediction_set = set(prediction.split(","))

            if label_set == prediction_set:
                match += 1
            if label_set.issubset(prediction_set):
                subset += 1
            total += 1
        print(f"Exact ACC: {match / total}, Subset ACC: {subset / total}")


def parse_result_for_prompt(input_file_path: str) -> None:
    match = 0
    total = 0
    text_and_result = {}
    with open(file=input_file_path, mode="r", encoding="utf-8") as fr:
        for line in fr:
            arr_line = line.strip().split("\t")
            prompt_text, label_name, label, generation_text = (
                arr_line[0],
                arr_line[1],
                arr_line[2],
                arr_line[3],
            )
            text = regex.search('这句用户评价:"(.*)"', prompt_text, regex.IGNORECASE).group(1)
            if text != "掉毛 第一次用掉了好多毛毛在脸上 差了 搞得皮肤都要过敏了":
                continue
            if text not in text_and_result:
                text_and_result[text] = {"labels": set(), "predictions": set()}
            if label.lower() == "a":
                text_and_result[text]["labels"].add(label_name)
            prediction = generation_text.split("-->")[1]
            if prediction.lower() == "a":
                text_and_result[text]["predictions"].add(label_name)
    for text, info in text_and_result.items():
        if info["labels"] == info["predictions"]:
            match += 1
        total += 1
    print(f"ACC: {match / total}")


if __name__ == "__main__":
    base_dir = (
        "/Users/bytedance/PycharmProjects/byte/EasyGuard/examples/fashion_gpt2/tasks/gpt2/ccr_experiments/data/corpus"
    )
    # generate_prompt_ner_data(
    #     input_train_file_path=os.path.join(base_dir, 'ner', 'im_hot_words_filter_sent_train_0228.json'),
    #     input_valid_file_path=os.path.join(base_dir, 'ner', 'im_hot_words_filter_sent_valid_0228.json'),
    #     train_out_file_path=os.path.join(base_dir, 'ner', 'ccr_ner_train.txt'),
    #     valid_out_file_path=os.path.join(base_dir, 'ner', 'ccr_ner_valid.jsonl'))

    # generate_prompt_mlc_data(input_train_file_path=os.path.join(base_dir, 'mlc', 'ccr_order_train.txt'),
    #                          input_valid_file_path=os.path.join(base_dir, 'mlc', 'ccr_order_valid.txt'),
    #                          train_out_file_path=os.path.join(base_dir, 'mlc', 'ccr_mlc_train.txt'),
    #                          valid_out_file_path=os.path.join(base_dir, 'mlc', 'ccr_mlc_valid.jsonl'))

    # convert_txt_to_parquet(input_file_path=os.path.join(base_dir, 'all', 'ccr_all_train.txt'),
    #                        output_file_path=os.path.join(base_dir, 'all', 'ccr_all_train.parquet'))
    # generate_prompt_multi_choice_data(input_train_file_path=os.path.join(base_dir, 'mlc', 'ccr_order_train.txt'),
    #                                   input_valid_file_path=os.path.join(base_dir, 'mlc', 'ccr_order_valid.txt'),
    #                                   train_out_file_path=os.path.join(base_dir, 'mlc',
    #                                                                    'ccr_multi_choice_train.parquet'),
    #                                   valid_out_file_path=os.path.join(base_dir, 'mlc', 'ccr_multi_choice_valid.jsonl'),
    #                                   label_map_file_path=os.path.join(base_dir, 'mlc', 'label_definition.txt'))
    # generate_train_prompt_data(input_file_path=os.path.join(data_base_dir, 'ccr_order_train.txt'),
    #                            label_definition_path='label_definition.txt',
    #                            parquet_out_file_path='ccr_train.parquet',
    #                            txt_out_file_path='ccr_train.txt')

    # generate_valid_prompt_data(input_file_path=os.path.join(data_base_dir, 'toy_valid.txt'),
    #                            label_definition_path='label_definition.txt',
    #                            valid_out_file_path='ccr_mlc_valid.jsonl')
    # parse_result_from_mlc(input_file_path='eval_detail.txt')
    parse_result_mlc_prompt(input_file_path="data/outputs/inference_result_2023032020.txt")
    # parse_result_ner_prompt(input_file_path='data/inference_result_2023031711.txt')
