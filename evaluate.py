import torch 
from torch.utils.data import DataLoader

from config import Config

from data.corpus import iter_text
from data.dataset import LMDataset

from tokenization.tokenizer import Tokenizer

from model.gpt import FawernGPT

from training.checkpoint import load_latest

def main():
    config = Config()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer.load(config.tokenizer_dir)

    ids = []
    for line in iter_text(config.data_path, lower_case=config.lower_case):
        ids.extend(tokenizer.encode(line, add_bos=False, add_eos=True, lower_case=config.lower_case))
    
    dataset = LMDataset(ids, block_size=config.block_size)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)

    model = FawernGPT(
        vocab_size=len(tokenizer.vocab.token_to_id),
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
        block_size=config.block_size,
    ).to(device)

    step = load_latest(config.save_dir, model, optimizer=None)
    model.eval()

    total_loss, count = 0.0, 0
    with torch.no_grad():
        for batch in dataloader:
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)

            _, loss = model(x, y)
            total_loss += loss.item()

            count += 1
    
    pseudo_ppl = float("inf") if count == 0 else pow(2.718281828, total_loss / max(1, count))
    print(f"Loaded step: {step} | Avg loss: {total_loss/max(1,count):.4f} | Pseudo-PPL: {ppl:.2f}")


if __name__ == "__main__":
    main()