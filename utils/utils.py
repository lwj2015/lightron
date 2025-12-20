import torch

@torch.no_grad()
def evaluate(model, tokenizer, eval_dataset, max_batches=10):
    model.eval()
    total_loss = 0
    steps = 0

    for i, batch in enumerate(eval_dataset):
        if i >= max_batches: break
        inputs = batch.to(model.device)  # 假设 batch 已经是 tensor
        logits = model(inputs)

        # 简单的 Next Token Prediction Loss
        # Shift logits and labels
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs[..., 1:].contiguous()

        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        total_loss += loss.item()
        steps += 1

    avg_loss = total_loss / steps
    perplexity = torch.exp(torch.tensor(avg_loss))
    model.train()
    return avg_loss, perplexity.item()
