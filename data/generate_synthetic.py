import torch
import numpy as np
from config import NUM_SAMPLES, SEQ_LEN, NUM_DEPTS, EMBED_DIM, MASK_PROB

def generate():
    seqs = torch.randint(0, NUM_DEPTS, (NUM_SAMPLES, SEQ_LEN))
    labels = torch.randint(0, 2, (NUM_SAMPLES, NUM_DEPTS)).float()
    season_ids = torch.randint(0, 4, (NUM_SAMPLES, SEQ_LEN))

    customer_emb = torch.randn(NUM_SAMPLES, EMBED_DIM)
    product_emb = torch.randn(NUM_SAMPLES, SEQ_LEN, EMBED_DIM)
    season_emb = torch.randn(NUM_SAMPLES, EMBED_DIM)
    weather_emb = torch.randn(NUM_SAMPLES, EMBED_DIM)
    promo_emb = torch.randn(NUM_SAMPLES, EMBED_DIM)

    return {
        "sequences": seqs,
        "season_ids": season_ids,
        "labels": labels,
        "customer_emb": customer_emb,
        "product_emb": product_emb,
        "season_emb": season_emb,
        "weather_emb": weather_emb,
        "promo_emb": promo_emb
    }