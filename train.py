import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List
import numpy as np
from settings import *
from data import TranslateData, collate_fn
from model import TranslateModel


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(reduction='none')
    with torch.no_grad():
        for batch in dataloader:
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
    return avg_loss


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = TranslateData('dataset/train_zh_idx.txt', 'dataset/train_en_idx.txt')
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: collate_fn(x, pad_id=PAD))
    valid_dataset = TranslateData('dataset/valid_zh_idx.txt', 'dataset/valid_en_idx.txt')
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: collate_fn(x, pad_id=PAD))

    vocab_size = VOCAB_SIZE
    embed_size = EMBED_SIZE
    hidden_size = HIDDEN_SIZE
    model = TranslateModel(vocab_size, embed_size, hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(reduction='none')

    best_valid_loss = float('inf')
    patience = PATIENCE
    patience_counter = 0
    num_epochs = MAX_EPOCH

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_tokens = 0
        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            optimizer.zero_grad()
            logits = model(input_ids)
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)
            loss_mask = loss_mask.view(-1)
            loss = criterion(logits, labels)
            masked_loss = (loss * loss_mask).sum() / (loss_mask.sum() + 1e-8)
            masked_loss.backward()
            optimizer.step()
            total_loss += masked_loss.item() * loss_mask.sum().item()
            total_tokens += loss_mask.sum().item()
        avg_train_loss = total_loss / (total_tokens + 1e-8)
        avg_valid_loss = evaluate(model, valid_dataloader, device)
        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, valid_loss={avg_valid_loss:.4f}")

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
            print("Best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pt")
            print(f"Model at epoch {epoch+1} saved.")


if __name__ == "__main__":
    train()