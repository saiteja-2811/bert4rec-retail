import torch
from config import *
from models.bert4rec import BERT4Rec
from data.generate_synthetic import generate
from train.loss_utils import get_loss
from train.trainer import train, evaluate

data = generate()
model = BERT4Rec(vocab_size=NUM_DEPTS, num_depts=NUM_DEPTS, emb_dim=EMBED_DIM, seq_len=SEQ_LEN, num_seasons=4)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = get_loss(pos_weights=[2.0, 1.5, 1.0, 1.0, 0.5])

train(model, data, optimizer, loss_fn, EPOCHS, BATCH_SIZE, GAMMA)
evaluate(model, data)