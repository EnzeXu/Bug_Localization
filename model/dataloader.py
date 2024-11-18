import random
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


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

    print("一共有多少行合理数据", len(valid_rows))
    # create tuple from valid rows
    br_len_list=[]
    for _, row in valid_rows.iterrows():
        title = str(row['issue_title'])
        body = str(row['issue_body'])
        br = title + "\t" + body

        br_len_list.append(len(br))

        method=str(row['commit_code_snippet'])
        score = int(row['relativity_score'])
        tuple = (br,method,score)
        tuple_data_list.append(tuple)

    # print("br的长度列表: ", sorted(br_len_list), "\n" )
    print("最小值：", np.min(br_len_list), "最大值：", np.max(br_len_list), "平均值：", np.mean(br_len_list), "中位数,", np.median(br_len_list) )
    return tuple_data_list


# tuple_data_list contains (br, method, score)
# Step 2: Shuffle and split the data to train, val and test set. input data is tuple_data_list
def split_data(tuple_list, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
    # random.shuffle(tuple_list)             #这句话暂时注释掉
    total = len(tuple_list)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)
    train_data_list = tuple_list[:train_end]
    valid_data_list = tuple_list[train_end:valid_end]
    test_data_list = tuple_list[valid_end:]
    return train_data_list, valid_data_list, test_data_list


# Step 2: Define the BLNT5 Dataset
class BLNT5Dataset(Dataset):
    def __init__(self, data, t5_tokenizer, code_t5_tokenizer, br_max_length=512, m_max_length=512):
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
def create_dataloader(data, t5_tokenizer, code_t5_tokenizer, batch_size=2):   #batch_size=2, 可以=16
    dataset = BLNT5Dataset(data, t5_tokenizer, code_t5_tokenizer)
    print("dataset length: ", len(dataset))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: {
            "br_input_ids": torch.stack([item["br_input_ids"] for item in x]),
            "br_attention_mask": torch.stack([item["br_attention_mask"] for item in x]),
            "method_input_ids": torch.stack([item["method_input_ids"] for item in x]),
            "method_attention_mask": torch.stack([item["method_attention_mask"] for item in x]),
            "score": torch.stack([item["score"] for item in x]),
        },
    )   #暂时让shuffle为false
    return dataloader
