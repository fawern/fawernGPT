import torch 
from config import Config
from tokenization.tokenizer import Tokenizer
from model.gpt import FawernGPT
from training.checkpoint import load_latest

def main():
    config = Config()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer.load(config.tokenizer_dir)

    model = FawernGPT(
        vocab_size=len(tokenizer.vocab.token_to_id),
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
        block_size=config.block_size,
    ).to(device)
    
    _ = load_latest(config.save_dir, model, optimizer=None)
    model.eval()

    prompt = "Hatta pek alelade, hiçbir hususiyeti olmayan, her gün etrafımızda yüzlercesini"
    x = torch.tensor([tokenizer.encode(prompt, add_bos=False, add_eos=False)], dtype=torch.long).to(device)

    with torch.no_grad():
        y = model.generate(x, max_new_tokens=90, temperature=1.0, top_k=50)
    
    txt = tokenizer.decode(y[0].tolist())
    print(f"PROMPT: {prompt}")
    print("----")
    print(txt)

if __name__ == "__main__":
    main()