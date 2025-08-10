import torch.nn as nn
import torch

class BERT4Rec(nn.Module):
    def __init__(self, vocab_size, num_depts, emb_dim, seq_len, num_seasons):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size + 1, emb_dim, padding_idx=vocab_size)
        self.season_pos_emb = nn.Embedding(num_seasons, emb_dim)
        self.pos_emb = nn.Embedding(seq_len, emb_dim)

        enc = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=2)

        self.struct_proj = nn.Sequential(
            nn.Linear(emb_dim * 4, emb_dim),
            nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Linear(emb_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_depts),
            nn.Sigmoid()
        )

    def forward(self, seqs, season_ids, customer_emb, product_emb, season_emb, weather_emb, promo_emb, decay=None):
        B, T = seqs.shape
        pos_ids = torch.arange(T, device=seqs.device).unsqueeze(0).expand(B, T)

        x = self.token_emb(seqs) + self.season_pos_emb(season_ids) + self.pos_emb(pos_ids)
        x = self.transformer(x)

        if decay is not None:
            x = (x * decay.unsqueeze(-1)).sum(1)
        else:
            x = x[:, 0, :]  # first token

        struct_input = torch.cat([season_emb, weather_emb, promo_emb, customer_emb], dim=-1)
        struct_vec = self.struct_proj(struct_input)
        out = self.head(torch.cat([x, struct_vec], dim=-1))
        return out