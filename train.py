import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from settings import *
from data import TranslateData, collate_fn
from model import TranslateModel
from report import report_progress_to_server
import time

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

def report(epoch: int, batch_id: int, total_batch: int):
    print(
        f"\rEpoch {epoch + 1} | Batch {batch_id + 1}/{total_batch} | ",
        end="",
    )

def get_gpu_mem_usage(device=None):
    if device is None:
        device = torch.cuda.current_device()
    mem_bytes = torch.cuda.memory_allocated(device)
    mem_mb = mem_bytes / 1024 / 1024
    return mem_mb

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_size = VOCAB_SIZE
    embed_size = EMBED_SIZE
    hidden_size = HIDDEN_SIZE
    model = TranslateModel(vocab_size, embed_size, hidden_size, 2).to(device)

    # Optional: Use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    train_dataset = TranslateData('dataset/train_zh_idx.txt', 'dataset/train_en_idx.txt')
    valid_dataset = TranslateData('dataset/valid_zh_idx.txt', 'dataset/valid_en_idx.txt')

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, pad_id=PAD),
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, pad_id=PAD),
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    print('Data Loaded. Train size:', len(train_dataset), 'Valid size:', len(valid_dataset))
    print('Total parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    print('Training started...')

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
        total_batch = len(train_dataloader)
        epoch_start_time = time.time()
        for batch_id, batch in enumerate(train_dataloader):
            batch_start_time = time.time()
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            loss_mask = batch["loss_mask"].to(device, non_blocking=True)

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
            batch_time = time.time() - batch_start_time
            if batch_id % 10 == 0:
                report(epoch, batch_id, total_batch)
                gpu_mem = get_gpu_mem_usage(device)
                report_progress_to_server(
                    epoch=epoch + 1,
                    loss=masked_loss.item(),
                    batch=batch_id + 1,
                    total_batches=total_batch,
                    lr=optimizer.param_groups[0]['lr'],
                    custom_metrics={
                        "gpu_mem": f"{gpu_mem:.2f}MB",
                        "batch_time": f"{batch_time:.3f}s"
                    }
                )
        epoch_time = time.time() - epoch_start_time
        avg_train_loss = total_loss / (total_tokens + 1e-8)
        avg_valid_loss = evaluate(model, valid_dataloader, device)
        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, valid_loss={avg_valid_loss:.4f}, epoch_time={epoch_time:.2f}s")
        report_progress_to_server(
            epoch=epoch + 1,
            loss=avg_train_loss,
            total_batches=total_batch,
            lr=optimizer.param_groups[0]['lr'],
            custom_metrics={
                "epoch_time": f"{epoch_time:.2f}s"
            }
        )
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
        if (epoch + 1) % 1 == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pt")
            print(f"Model at epoch {epoch+1} saved.")

if __name__ == "__main__":
    train()