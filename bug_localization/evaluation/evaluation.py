import torch
import pickle
import torch.nn as nn
from tqdm import tqdm

from ..dataset import set_random_seed
from .metrics import metric_accuracy, metric_precision, metric_recall, metric_f1_score
from ..model.dataloader import process_csv_to_tuple_list, split_data, create_dataloader
from ..utils.pretrained import T5CODE_TOKENIZER, T5TEXT_TOKENIZER
from ..model import BLNT5


def load_model(model, model_load_path):
    save_path = model_load_path  # f"{save_model_folder}/best_model.pth"
    checkpoint = torch.load(save_path, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])

    if "train_loss" in checkpoint:
        epoch = checkpoint["epoch"]
        epochs = checkpoint["epochs"]
        train_loss = checkpoint["train_loss"]
        val_loss = checkpoint["val_loss"]
        timestring = checkpoint["timestring"]
        # lr = checkpoint["lr"]
        # seed = checkpoint["seed"]

        print(f"[{timestring}] Model loaded successfully from epoch {epoch}/{epochs} with train loss {train_loss:.4f} and val loss {val_loss:.4f}.")
    else:
        timestring = checkpoint["timestring"]
        print(f"[{timestring}] This is a random weights without training")

    return model


def test_evaluation(model_load_path, data_path, timestring=None):
    print("#" * 200)
    set_random_seed(42)

    t5_tokenizer = T5TEXT_TOKENIZER.from_pretrained("google-t5/t5-small", legacy=True)
    code_t5_tokenizer = T5CODE_TOKENIZER.from_pretrained("Salesforce/codet5-small")  # Example CodeT5 model

    gpu_id = 3
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    print(f"Using device: {device}")

    batch_size = 32

    with open(data_path, "rb") as f:
        test_data_list = pickle.load(f)
    score_list = [item[2] for item in test_data_list]
    print(f"Score distribution [Truth]: 0 count = {score_list.count(0)}, 1 count = {score_list.count(1)}, Total = {len(score_list)}")

    test_loader = create_dataloader(test_data_list, t5_tokenizer, code_t5_tokenizer, batch_size=batch_size,
                                    shuffle=False, name="test")

    model = BLNT5(fix_pretrain_weights=True)
    model.to(device)
    model = load_model(model, model_load_path)
    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    all_predictions = []
    all_truths = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(test_loader):
            br_input_ids = batch["br_input_ids"].to(device)
            br_attention_mask = batch["br_attention_mask"].to(device)

            method_input_ids = batch["method_input_ids"].to(device)
            method_attention_mask = batch["method_attention_mask"].to(device)

            targets = batch["score"].to(device)

            # Forward pass
            outputs = model(br_input_ids, br_attention_mask, method_input_ids, method_attention_mask)
            # print(f"outputs.squeeze(1): {outputs.squeeze(1)}")
            # print(f"targets: {targets}")
            loss = criterion(outputs.squeeze(1), targets)
            total_loss += loss.item()

            sigmoid_outputs = torch.sigmoid(outputs)
            binary_predictions = (sigmoid_outputs > 0.5).long()

            all_predictions.extend(binary_predictions.cpu().numpy())
            all_truths.extend(targets.cpu().numpy())

            # break

    avg_loss = total_loss / len(test_loader)
    print(f"Score distribution [Truth]: 0 count = {all_truths.count(0)}, 1 count = {all_truths.count(1)}, Total = {len(all_truths)}")
    print(f"Score distribution [Prediction]: 0 count = {all_predictions.count(0)}, 1 count = {all_predictions.count(1)}, Total = {len(all_predictions)}")

    overall_accuracy, _, _ = metric_accuracy(all_truths, all_predictions)
    overall_precision, _, _ = metric_precision(all_truths, all_predictions)
    overall_recall, _, _ = metric_recall(all_truths, all_predictions)
    overall_f1_score = metric_f1_score(all_truths, all_predictions)
    print(f"Timestring: {timestring}")
    print(f"Model weights source: {model_load_path}")
    print(f"Loss (Cross-Entropy): {avg_loss:.12f}")
    print(f"Accuracy: {overall_accuracy:.12f}")
    print(f"Precision: {overall_precision:.12f}")
    print(f"Recall: {overall_recall:.12f}")
    print(f"F1 Score: {overall_f1_score:.12f}")
    print(f"{timestring},{model_load_path},{all_truths.count(0)},{all_truths.count(1)},{len(all_truths)},"
          f"{all_predictions.count(0)},{all_predictions.count(1)},{len(all_predictions)},"
          f"{avg_loss},{overall_accuracy},{overall_precision},{overall_recall},{overall_f1_score}")
    with open("log_16_20.csv", "a") as f:
        f.write(f"{timestring},{model_load_path},{all_truths.count(0)},{all_truths.count(1)},{len(all_truths)},"
                f"{all_predictions.count(0)},{all_predictions.count(1)},{len(all_predictions)},"
                f"{avg_loss},{overall_accuracy},{overall_precision},{overall_recall},{overall_f1_score}\n")
    print()


if __name__ == "__main__":
    time_string_list = [
        # "20241125_144153_182984",
        # "20241125_144223_713220",
        # "20241125_144254_060896",
        # "20241125_144318_994388",
        # "20241125_144341_308837",
        # "20241125_151709_404653",  # 6
        # "20241125_151817_922505",  # 7
        # "20241125_151840_380934",  # 8
        # "20241125_151910_970700",  # 9
        # "20241125_151952_270729",  # 10
        # "20241125_152801_624755",  # 1
        # "20241125_152834_619761",  # 2
        # "20241125_152901_962635",  # 3
        # "20241125_152929_369774",  # 4
        # "20241125_152957_614023",  # 5
        # "20241125_155411_720225",  # 11
        # "20241125_155440_415618",  # 12
        # "20241125_155503_376250",  # 13
        # "20241125_155526_537369",  # 14
        # "20241125_155549_759435",  # 15
        # "20241125_155839_683878",  # 16
        # "20241125_155908_022633",  # 17
        # "20241125_155936_046389",  # 18
        # "20241125_160003_760665",  # 19
        # "20241125_160032_203238",  # 20
        # "20241125_152342_633071",  # random seed: 42, batch size: 64
        # "20241125_162613_488842",  # random seed: 99, batch size 64 best!
        "20241125_182221_685514",  # random seed: 99, batch size 64 scheduler SGD
        "20241125_182411_769566",  # random seed: 99, batch size 64 scheduler Adam

    ]
    for one_time_string in time_string_list:
        test_evaluation(f"save_model/{one_time_string}/best_model.pth", f"save_model/{one_time_string}/test_data.pkl", timestring=one_time_string)
    pass
