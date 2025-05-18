import torch
from settings import *
from data import load_bpe_model
from model import TranslateModel

def beam_search_generate(input_sentence, src_lang=ZH, tgt_lang=EN, max_len=100, beam_width=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = load_bpe_model('dataset/bpe.model')
    input_ids = [src_lang] + sp.encode(input_sentence, out_type=int) + [tgt_lang]

    vocab_size = VOCAB_SIZE
    embed_size = EMBED_SIZE
    hidden_size = HIDDEN_SIZE
    model = TranslateModel(vocab_size, embed_size, hidden_size, 2).to(device)
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.eval()

    # Each beam is (tokens, score)
    beams = [(input_ids.copy(), 0.0)]
    completed = []

    for _ in range(max_len):
        new_beams = []
        for tokens, score in beams:
            input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(input_tensor)
            next_token_logits = logits[0, -1, :]
            log_probs = torch.log_softmax(next_token_logits, dim=-1)
            topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)

            for log_prob, idx in zip(topk_log_probs.tolist(), topk_indices.tolist()):
                new_tokens = tokens + [idx]
                new_score = score + log_prob
                if idx == EOS:
                    completed.append((new_tokens, new_score))
                else:
                    new_beams.append((new_tokens, new_score))
        # Keep top beam_width beams
        print(beams)
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        # If all beams ended with EOS, stop early
        if not beams:
            break

    # Add remaining beams to completed if not already ended with EOS
    completed += beams
    # Choose the best completed sequence
    best_tokens, best_score = max(completed, key=lambda x: x[1])

    try:
        start = best_tokens.index(tgt_lang) + 1
        end = best_tokens.index(EOS) if EOS in best_tokens else len(best_tokens)
        output_ids = best_tokens[start:end]
    except ValueError:
        output_ids = best_tokens

    output_text = sp.decode(output_ids)
    return output_text

if __name__ == "__main__":
    input_sentence = input()  # Example Chinese input
    translation = beam_search_generate(input_sentence, src_lang=EN, tgt_lang=ZH)
    print("Translation:", translation)