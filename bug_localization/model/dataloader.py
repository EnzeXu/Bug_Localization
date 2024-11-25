import random
import pandas as pd
import numpy as np

import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from ..utils import T5TEXT_TOKENIZER, T5CODE_TOKENIZER


#Step1: read from csv file to the get the tuple_data_list=[], which contains all tuples(br, m, score), return tuple_data_list
# 创造三元组的过程中，如果O列不是0/1，跳过。然后：先输出br的长度，然后再有一个筛选，长度不在一定范围内的不要
def process_csv_to_tuple_list(data_path):
    tuple_data_list = []

    # Read the CSV file with proper encoding and error handling
    try:
        df = pd.read_csv(data_path, encoding='latin1', low_memory=False)
    except Exception as e:
        print(f"Error reading the file: {e}")
        return []

    # Ensure 'relativity_score' is numeric(not null) and filter valid rows
    df['relativity_score'] = pd.to_numeric(df['relativity_score'], errors='coerce')
    valid_rows = df[ (~df['relativity_score'].isnull())  & (~df['issue_title'].isnull()) & (~df['issue_body'].isnull()) & (~df['commit_code_snippet'].isnull()) & (df['relativity_score'].isin([0, 1])) ]

    print("How many valid rows:", len(valid_rows))
    # create tuple from valid rows
    br_token_len_list = []
    method_token_len_list = []
    score_list = []
    text_tokenizer = T5TEXT_TOKENIZER.from_pretrained("google-t5/t5-small", legacy=True)
    code_tokenizer = T5CODE_TOKENIZER.from_pretrained("Salesforce/codet5-small")
    br_thres_low, br_thres_high = 0,355#32, 128
    m_thres_low, m_thres_high = 0,229#32, 128
    # print("Parsing data from csv:")
    for idx, row in tqdm(valid_rows.iterrows(), total=len(valid_rows), desc="Processing rows"):  # for idx, row in valid_rows.iterrows():
        # if idx == 2:
        #     break
        title = str(row['issue_title']).strip()
        body = str(row['issue_body']).strip()
        br = title + "\t" + body
        method = str(row['commit_code_snippet'])

        br_input_ids = text_tokenizer(br, return_tensors="pt").input_ids
        method_input_ids = code_tokenizer(method, return_tensors="pt").input_ids
        # print(len(input_ids[0]))

        # br_tokens = text_tokenizer(
        #     br.replace("\n", " ").replace("\t", " ").strip(),
        #     max_length=512,
        #     truncation=True,
        #     padding="max_length",
        #     return_tensors="pt"
        # )
        # print(f"br token length: {len(br_input_ids[0])} method token length: {len(method_input_ids[0])}")

        if len(br_input_ids[0]) > br_thres_high or len(br_input_ids[0]) < br_thres_low:
            continue
        if len(method_input_ids[0]) > m_thres_high or len(method_input_ids[0]) < m_thres_low:
            continue
        # print(f"[passed] br token length: {len(br_input_ids[0])} method token length: {len(method_input_ids[0])}")
        # print(f"br: {br}")
        # print(f"BR: '{br}'")

        br_token_len_list.append(len(br_input_ids[0]))
        method_token_len_list.append(len(method_input_ids[0]))

        score = int(row['relativity_score'])
        score_list.append(score)
        data_tuple = (br, method, score)
        tuple_data_list.append(data_tuple)

    np.save("br_length.npy", np.array(br_token_len_list))
    np.save("method_length.npy", np.array(method_token_len_list))
    np.save("score.npy", np.array(score_list))
    print(f"Score distribution: 0 count = {score_list.count(0)}, 1 count = {score_list.count(1)}, Total = {len(score_list)}")
    # print(br_len_list)

    # print("br的长度列表: ", sorted(br_len_list), "\n" )
    print(f"Bug Report Statistics (length={len(br_token_len_list)}):")
    print("min：", np.min(br_token_len_list), "max：", np.max(br_token_len_list), "mean：", np.mean(br_token_len_list), "median,", np.median(br_token_len_list))
    print(f"Method Statistics (length={len(method_token_len_list)}):")
    print("min：", np.min(method_token_len_list), "max：", np.max(method_token_len_list), "mean：", np.mean(method_token_len_list), "median,", np.median(method_token_len_list))
    return tuple_data_list


# tuple_data_list contains (br, method, score)
# Step 2: Shuffle and split the data to train, val and test set. input data is tuple_data_list
def split_data(tuple_list, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
    random.shuffle(tuple_list)             #这句话暂时注释掉
    total = len(tuple_list)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)
    train_data_list = tuple_list[:train_end]
    # print(f"train_data_list[:2]: {train_data_list[:2]}")
    valid_data_list = tuple_list[train_end:valid_end]
    test_data_list = tuple_list[valid_end:]
    return train_data_list, valid_data_list, test_data_list


# Step 2: Define the BLNT5 Dataset
class BLNT5Dataset(Dataset):
    def __init__(self, data, t5_tokenizer, code_t5_tokenizer, br_max_length=355, m_max_length=229):
        self.data = data
        self.t5_tokenizer = t5_tokenizer
        self.code_t5_tokenizer = code_t5_tokenizer

        self.br_max_length = br_max_length
        self.m_max_length = m_max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        br, method, score = self.data[idx]

        # Tokenize `br` using T5 tokenizer
        br_tokens = self.t5_tokenizer(
            br.replace("\n", " ").replace("\t", " ").strip(),
            max_length=self.br_max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        # Tokenize `method` using CodeT5 tokenizer
        method_tokens = self.code_t5_tokenizer(
            method,
            max_length=self.m_max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        # Convert score to tensor
        score_tensor = torch.tensor(score, dtype=torch.float)

        return {
            "br_input_ids": br_tokens["input_ids"].squeeze(0),
            "br_attention_mask": br_tokens["attention_mask"].squeeze(0),
            "method_input_ids": method_tokens["input_ids"].squeeze(0),
            "method_attention_mask": method_tokens["attention_mask"].squeeze(0),
            "score": score_tensor,
        }


# Step 2: Prepare DataLoader
def create_dataloader(data, t5_tokenizer, code_t5_tokenizer, batch_size, shuffle=False, name=None):   #batch_size=2, 可以=16
    dataset = BLNT5Dataset(data, t5_tokenizer, code_t5_tokenizer)
    print(f"{name + ' ' if name else ''}dataset length: ", len(dataset))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda x: {
            "br_input_ids": torch.stack([item["br_input_ids"] for item in x]),
            "br_attention_mask": torch.stack([item["br_attention_mask"] for item in x]),
            "method_input_ids": torch.stack([item["method_input_ids"] for item in x]),
            "method_attention_mask": torch.stack([item["method_attention_mask"] for item in x]),
            "score": torch.stack([item["score"] for item in x]),
        },
    )   #暂时让shuffle为false
    return dataloader
