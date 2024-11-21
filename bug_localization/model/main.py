import os.path

from .dataloader import process_csv_to_tuple_list, split_data, create_dataloader
from ..utils.pretrained import T5CODE_TOKENIZER, T5TEXT_TOKENIZER
from .model import BLNT5

import torch
import time
import torch.nn as nn
import torch.optim as optim
import wandb

from tqdm import tqdm

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
        # break

    avg_loss = total_loss / len(train_loader)
    # print(f"Train Loss: {avg_loss:.4f}")
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
            # print(f"outputs.squeeze(1): {outputs.squeeze(1)}")
            # print(f"targets: {targets}")
            loss = criterion(outputs.squeeze(1), targets)
            total_loss += loss.item()
            # break

    avg_loss = total_loss / len(valid_loader)
    # print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss

def run(model, train_loader, valid_loader, criterion, optimizer, device, epochs, main_path):
    best_val_loss = float("inf")

    save_model_folder = os.path.join(main_path, "save_model")
    if not os.path.exists(save_model_folder):
        os.makedirs(save_model_folder)

    t0 = time.time()
    t_prev = t0
    epoch_step = 1  # 10

    for epoch in range(epochs):
        # print(f"Epoch {epoch + 1}/{epochs} start")
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, valid_loader, criterion, device)
        t_tmp = time.time()

        wandb.log({"epoch": epoch, "train_loss": train_loss, "valid_loss": val_loss})

        if (epoch + 1) % epoch_step == 0:
            print(f"Epoch {epoch + 1:04d}/{epochs:04d}, train_loss: {train_loss:.12f}, val_loss: {val_loss:.12f}, "
                  f"lr: {optimizer.param_groups[0]['lr']:.6f} "
                  f"(t_cost: {t_tmp - t_prev:.1f} s, "
                  f"t_total: {(t_tmp - t0) / 60:.3f} min, "
                  f"t_remain: {(t_tmp - t0) / 60 / (epoch + 1) * (epochs - epoch - 1):.3f} min)", end="")
            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"{save_model_folder}/best_model.pth")
                print(" [Current Best]")
                # print()
            else:
                print()

    print("Training complete.")


def main_run(main_path):
    wandb.init(project="BLNT5", name="In-Dis")
    # step1: 数据处理
    # 从robolectric_dataset.csv中读取数据，创造三元组列表。
    data_path = os.path.join(main_path, "data/csv/robolectric@robolectric.csv")
    tuple_list = process_csv_to_tuple_list(data_path)
    print("# of tuple (total):", len(tuple_list))  # 22527
    # for (br, m, score) in tuple_list:
    #     print("(br=", br,"  m= ", m, "  score=", score, ")\n")

    # step2: split the data into train, val, and test set, (8:1:1) and use dataloader to become the input to the model
    train_data_list, valid_data_list, test_data_list = split_data(tuple_list)
    print("train length：", len(train_data_list), "val length：", len(valid_data_list), "test length：",
          len(test_data_list))  # 18021，2252，2254

    t5_tokenizer = T5TEXT_TOKENIZER.from_pretrained("t5-small")
    code_t5_tokenizer = T5CODE_TOKENIZER.from_pretrained("Salesforce/codet5-small")  # Example CodeT5 model

    batch_size = 64

    train_loader = create_dataloader(train_data_list, t5_tokenizer, code_t5_tokenizer, batch_size=batch_size, shuffle=True)
    valid_loader = create_dataloader(valid_data_list, t5_tokenizer, code_t5_tokenizer, batch_size=batch_size, shuffle=False)
    test_loader = create_dataloader(test_data_list, t5_tokenizer, code_t5_tokenizer, batch_size=batch_size, shuffle=False)
    print("train_loader length：", len(train_loader), "valid_load length：", len(valid_loader), "test_loader length：",
          len(test_loader))  # 9011, 1126, 1127

    # for batch in test_loader:
    #     print( "Bug report tokens:",  batch["br_input_ids"], "Bug Report Input IDs Shape:", batch["br_input_ids"].shape)
    #     print( "br_attention_mask",  batch["br_attention_mask"], "br_attention_mask Shape:", batch["br_attention_mask"].shape)
    #
    #     print( "Method Input tokens:", batch["method_input_ids"], "Method Input IDs Shape:", batch["method_input_ids"].shape)
    #     print( "method_attention_mask", batch["method_attention_mask"], "method_attention_mask Shape:", batch["method_attention_mask"].shape)
    #
    #     print( "score", batch["score"], "Score Tensor Shape:", batch["score"].shape)
    #     break

    # step1和step2都在dataloader.py中的函数完成

    # step3: init model
    model = BLNT5()
    # print(model)
    gpu_id = 3

    # Check if CUDA is available
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    # Set the default GPU for computation (optional)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)

    print(f"Using device: {device}")
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
    optimizer = optim.Adam(model.parameters(), lr=1e-3)


    # Run training and validation
    run(model, train_loader, valid_loader, criterion, optimizer, device, epochs=10, main_path=main_path)
    wandb.finish()


if __name__ == '__main__':
    main_run(main_path="./")
