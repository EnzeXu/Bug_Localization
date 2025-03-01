import os
import argparse
import torch
import time
import torch.nn as nn
import torch.optim as optim
import wandb
import pickle
from tqdm import tqdm

from ..dataset import get_now_string

from .dataloader import process_csv_to_tuple_list, split_data, create_dataloader
from ..utils.pretrained import T5CODE_TOKENIZER, T5TEXT_TOKENIZER
from ..dataset import get_now_string, set_random_seed
from .model import BLNT5Concat, BLNT5Cosine


#train the data for 1 epoch using training dataset


def save_random_weight(model, save_model_folder, timestring, seed=None):
    save_dic = {
        "timestring": timestring,
        "state_dict": model.state_dict(),
        "model": model.__class__.__name__,
        "seed": seed,
    }
    torch.save(save_dic, f"{save_model_folder}/random.pth")


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(train_loader):
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
        for batch in tqdm(valid_loader):
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

def run(model, train_loader, valid_loader, criterion, device, optimizer, scheduler, epochs, main_path, timestring, seed=None):
    best_val_loss = float("inf")

    save_model_folder = os.path.join(main_path, "save_model", timestring)
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

        wandb.log({"epoch": epoch, "train_loss": train_loss, "valid_loss": val_loss, "lr": optimizer.param_groups[0]['lr']})

        if (epoch + 1) % epoch_step == 0:
            print(f"[{get_now_string()}] Epoch {epoch + 1:04d}/{epochs:04d}, train_loss: {train_loss:.12f}, val_loss: {val_loss:.12f}, "
                  f"lr: {optimizer.param_groups[0]['lr']:.6f} "
                  f"(t_cost: {t_tmp - t_prev:.1f} s, "
                  f"t_total: {(t_tmp - t0) / 60:.3f} min, "
                  f"t_remain: {(t_tmp - t0) / 60 / (epoch + 1) * (epochs - epoch - 1):.3f} min)", end="")
            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_dic = {
                    "timestring": timestring,
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "epochs": epochs,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": optimizer.param_groups[0]['lr'],
                    "seed": seed,
                    "model": model.__class__.__name__,
                }
                torch.save(save_dic, f"{save_model_folder}/best_model.pth")
                print(" [Current Best]")
                # print()
            else:
                print()

        if epoch + 1 == epochs:
            save_dic = {
                "timestring": timestring,
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "epochs": epochs,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": optimizer.param_groups[0]['lr'],
                "seed": seed,
                "model": model.__class__.__name__,
            }
            torch.save(save_dic, f"{save_model_folder}/last_model.pth")
        if scheduler is not None:
            scheduler.step()

    print("Training complete.")


def main_run(main_path):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=42, help="Set random seed"
    )
    parser.add_argument(
        "--gpu_id", type=int, default=3, help="Set GPU ID (default:3)"
    )
    parser.add_argument(
        "--random", action="store_true", default=False, help="Enable random weights (skip training, data save and model save)"
    )
    parser.add_argument(
        "--no_wandb", action="store_true", default=False, help="Disable wandb"
    )
    parser.add_argument(
        "--model", type=str, default="BLNT5Concat", choices=["BLNT5Concat", "BLNT5Cosine"], help="BLNT5Concat or BLNT5Cosine"
    )
    args = parser.parse_args()
    if args.random:
        args.no_wandb = True

    timestring = get_now_string()
    seed = args.seed
    if not args.no_wandb:
        wandb.init(project=args.model, name=f"In-Dis_{timestring}")
    set_random_seed(seed)
    print("#" * 200)
    print(f"[## Timestring: {timestring} ##]")
    print("#" * 200)
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

    t5_tokenizer = T5TEXT_TOKENIZER.from_pretrained("google-t5/t5-small", legacy=True)
    code_t5_tokenizer = T5CODE_TOKENIZER.from_pretrained("Salesforce/codet5-small")  # Example CodeT5 model

    batch_size = 64
    print("#" * 200)
    print(f"train_data_list[0]:\n{train_data_list[0]}")
    print(f"valid_data_list[0]:\n{valid_data_list[0]}")
    print(f"test_data_list[0]:\n{test_data_list[0]}")
    print("#" * 200)

    save_model_folder = os.path.join(main_path, "save_model", timestring)
    if not os.path.exists(save_model_folder):
        os.makedirs(save_model_folder)

    if not args.random:
        with open(os.path.join(save_model_folder, "train_data.pkl"), "wb") as f:
            pickle.dump(train_data_list, f)
        with open(os.path.join(save_model_folder, "valid_data.pkl"), "wb") as f:
            pickle.dump(valid_data_list, f)
    with open(os.path.join(save_model_folder, "test_data.pkl"), "wb") as f:
        pickle.dump(test_data_list, f)

    train_loader = create_dataloader(train_data_list, t5_tokenizer, code_t5_tokenizer, batch_size=batch_size, shuffle=True, name="train")
    valid_loader = create_dataloader(valid_data_list, t5_tokenizer, code_t5_tokenizer, batch_size=batch_size, shuffle=False, name="valid")
    test_loader = create_dataloader(test_data_list, t5_tokenizer, code_t5_tokenizer, batch_size=batch_size, shuffle=False, name="test")
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
    if args.model == "BLNT5Concat":
        model = BLNT5Concat(fix_pretrain_weights=True)
    else:
        model = BLNT5Cosine(fix_pretrain_weights=True)
    # print(model)
    gpu_id = args.gpu_id

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
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.SGD(model.parameters(), lr=0.1)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    # Run training and validation

    save_random_weight(model, save_model_folder, timestring, seed)

    if not args.random:
        run(model, train_loader, valid_loader, criterion, device, optimizer, scheduler=None, epochs=100, main_path=main_path, timestring=timestring, seed=seed)
    if not args.no_wandb:
        wandb.finish()


if __name__ == '__main__':
    main_run(main_path="./")
