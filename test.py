import torch
from settings import *
from data import TranslateData, collate_fn
from model import TranslateModel

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = VOCAB_SIZE
    embed_size = EMBED_SIZE
    hidden_size = HIDDEN_SIZE
    model = TranslateModel(vocab_size, embed_size, hidden_size, 2).to(device)
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.eval()

    test_dataset = TranslateData('dataset/test_zh_idx.txt', 'dataset/test_en_idx.txt')
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, pad_id=PAD)
    )

    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            logits = model(input_ids)
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)
            loss_mask = loss_mask.view(-1)
            loss = criterion(logits, labels)
            masked_loss = (loss * loss_mask).sum()
            total_loss += masked_loss.item()
            total_tokens += loss_mask.sum().item()
    avg_loss = total_loss / (total_tokens + 1e-8)
    print(f"Test loss: {avg_loss:.4f}")

if __name__ == "__main__":
    test()