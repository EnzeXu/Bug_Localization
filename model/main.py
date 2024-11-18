from dataloader import process_csv_to_tuple_list, split_data, create_dataloader
from pretrained import T5CODE_TOKENIZER, T5TEXT_TOKENIZER
from model import BLNT5

import torch
import torch.nn as nn
import torch.optim as optim

#train the data for 1 epoch using training dataset
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        br_input_ids = batch["br_input_ids"].to(device)
        br_attention_mask = batch["br_attention_mask"].to(device)

        method_input_ids = batch["method_input_ids"].to(device)
        method_attention_mask = batch["method_attention_mask"].to(device)

        targets = batch["score"].to(device)

        # Forward pass
        logits = model(br_input_ids, br_attention_mask, method_input_ids, method_attention_mask)
        loss = criterion(logits.squeeze(1), targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        break

    avg_loss = total_loss / len(train_loader)
    print(f"Train Loss: {avg_loss:.4f}")
    return avg_loss


#evaluate the model on the validation or test dataset without updating weights
def validate(model, valid_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in valid_loader:
            br_input_ids = batch["br_input_ids"].to(device)
            br_attention_mask = batch["br_attention_mask"].to(device)

            method_input_ids = batch["method_input_ids"].to(device)
            method_attention_mask = batch["method_attention_mask"].to(device)

            targets = batch["score"].to(device)

            # Forward pass
            outputs = model(br_input_ids,br_attention_mask,method_input_ids,method_attention_mask)
            loss = criterion(outputs.squeeze(1), targets)
            total_loss += loss.item()
            break

    avg_loss = total_loss / len(valid_loader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss

def run(model, train_loader, valid_loader, criterion, optimizer, device, epochs=1):
    best_val_loss = float("inf")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs} start")
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, valid_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{epochs}, train_loss: {train_loss}, valid_loss: {val_loss}" )

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "save_model/best_model.pth")
            print("Current best model saved!")

    print("Training complete.")


if __name__ == '__main__':
    # step1: 数据处理
    # 从robolectric_dataset.csv中读取数据，创造三元组列表。
    data_path="../data/robolectric_dataset.csv"
    tuple_list = process_csv_to_tuple_list(data_path)
    print("一共有多少tuple数据", len(tuple_list))    # 22527
    # for (br, m, score) in tuple_list:
    #     print("(br=", br,"  m= ", m, "  score=", score, ")\n")

    # step2: split the data into train, val, and test set, (8:1:1) and use dataloader to become the input to the model
    train_data_list, valid_data_list, test_data_list = split_data(tuple_list)
    print("训练集的长度是：", len(train_data_list), "验证集的长度是：", len(valid_data_list),"测试集的长度是：",len(test_data_list) )    # 18021，2252，2254


    t5_tokenizer = T5TEXT_TOKENIZER.from_pretrained("t5-small")
    code_t5_tokenizer = T5CODE_TOKENIZER.from_pretrained("Salesforce/codet5-small")  # Example CodeT5 model

    train_loader = create_dataloader(train_data_list, t5_tokenizer, code_t5_tokenizer)
    valid_loader = create_dataloader(valid_data_list, t5_tokenizer, code_t5_tokenizer)
    test_loader = create_dataloader(test_data_list, t5_tokenizer, code_t5_tokenizer)
    print( "train_loader的长度是：", len(train_loader), "valid_loader的长度是：", len(valid_loader),"test_loader的长度是：",len(test_loader) )   # 9011, 1126, 1127

    # for batch in test_loader:
    #     print( "Bug report tokens:",  batch["br_input_ids"], "Bug Report Input IDs Shape:", batch["br_input_ids"].shape)
    #     print( "br_attention_mask",  batch["br_attention_mask"], "br_attention_mask Shape:", batch["br_attention_mask"].shape)
    #
    #     print( "Method Input tokens:", batch["method_input_ids"], "Method Input IDs Shape:", batch["method_input_ids"].shape)
    #     print( "method_attention_mask", batch["method_attention_mask"], "method_attention_mask Shape:", batch["method_attention_mask"].shape)
    #
    #     print( "score", batch["score"], "Score Tensor Shape:", batch["score"].shape)
    #     break

    #step1和step2都在dataloader.py中的函数完成

    # step3: init model
    model = BLNT5()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # one forward pass to test
    # for batch in train_loader:
    #     br_input_ids = batch["br_input_ids"]
    #     br_attention_mask = batch["br_attention_mask"]
    #     method_input_ids = batch["method_input_ids"]
    #     method_attention_mask = batch["method_attention_mask"]
    #
    #     outputs = model(br_input_ids,br_attention_mask, method_input_ids, method_attention_mask)
    #
    #     print(" outputs of 1 batch for model test:", outputs, "shape: ", outputs.shape)     # (2)
    #     break

    # step4: run()

    # Define criterion and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Run training and validation
    run(model, train_loader, valid_loader, criterion, optimizer, device)
