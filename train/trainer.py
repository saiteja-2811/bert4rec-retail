from sklearn.metrics import f1_score
import torch
import torch.nn.functional as F

def decay_weights(seq_len, gamma):
    return torch.tensor([gamma**i for i in reversed(range(seq_len))], dtype=torch.float)

def train(model, data, optimizer, loss_fn, epochs, batch_size, gamma):
    dataset = torch.utils.data.TensorDataset(
        data['sequences'], data['season_ids'], data['labels'],
        data['customer_emb'], data['product_emb'],
        data['season_emb'], data['weather_emb'], data['promo_emb']
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in loader:
            seqs, season_ids, labels, cust, prod, seas, weath, promo = batch
            decay = decay_weights(seqs.size(1), gamma).unsqueeze(0).to(seqs.device)

            out = model(seqs, season_ids, cust, prod, seas, weath, promo, decay)
            loss = loss_fn(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        decay = decay_weights(data['sequences'].size(1), gamma=0.95).unsqueeze(0)
        preds = model(
            data['sequences'], data['season_ids'],
            data['customer_emb'], data['product_emb'],
            data['season_emb'], data['weather_emb'],
            data['promo_emb'], decay
        )
        preds_bin = (preds > 0.5).float()
        f1 = f1_score(data['labels'].numpy(), preds_bin.numpy(), average='macro')
        print(f"Macro F1: {f1:.4f}")