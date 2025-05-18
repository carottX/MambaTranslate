import torch
from settings import *
from data import load_bpe_model
from model import TranslateModel

def generate(input_sentence, src_lang=ZH, tgt_lang=EN, max_len=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load BPE model
    sp = load_bpe_model('dataset/bpe.model')
    # Encode input
    input_ids = [src_lang] + sp.encode(input_sentence, out_type=int) + [tgt_lang]
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

    # Load model
    vocab_size = VOCAB_SIZE
    embed_size = EMBED_SIZE
    hidden_size = HIDDEN_SIZE
    model = TranslateModel(vocab_size, embed_size, hidden_size, 2).to(device)
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.eval()

    # Start generation
    generated = input_ids.copy()
    for _ in range(max_len):
        input_tensor = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_tensor)
        next_token_logits = logits[0, -1, :]
        next_token = torch.argmax(next_token_logits).item()
        generated.append(next_token)
        if next_token == EOS:
            break

    # Only decode tokens after tgt_lang and before EOS
    try:
        start = generated.index(tgt_lang) + 1
        end = generated.index(EOS) if EOS in generated else len(generated)
        output_ids = generated[start:end]
    except ValueError:
        output_ids = generated

    output_text = sp.decode(output_ids)
    return output_text

if __name__ == "__main__":
    input_sentence = "你好，世界！"  # Example Chinese input
    translation = generate(input_sentence, src_lang=ZH, tgt_lang=EN)
    print("Translation:", translation)