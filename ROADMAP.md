# Neuva Roadmap

What's planned after v1.0.0.

---

## v1.1 — Recurrent Layers
- `layer rnn(hidden_size, relu)` — vanilla RNN
- `layer lstm(hidden_size)` — Long Short-Term Memory
- `layer gru(hidden_size)` — Gated Recurrent Unit
- Sequence data loader with padding and masking support

## v1.2 — Custom Loss Functions
- `loss fn my_loss(pred, target) { return ... }` syntax
- Built-in Huber, focal, contrastive, and triplet losses
- Weighted cross-entropy for imbalanced datasets

## v1.3 — Multi-GPU Training
- Automatic `DataParallel` / `DistributedDataParallel` wrapping
- `train Model on data for N epochs, devices = [0, 1]`
- Gradient accumulation for large effective batch sizes

## v1.4 — Transformer and Attention Layers
- `layer attention(heads=8, dim=512)` — multi-head self-attention
- `layer transformer(heads=8, dim=512, ff=2048)` — full encoder block
- Positional encoding built-in

## v1.5 — Web Playground
- Browser-based Neuva editor with syntax highlighting
- Live execution against a sandboxed backend
- Share programs as short URLs

## v2.0 — Community Model Registry
- `neuva publish MyModel` — share trained architectures
- `let model = fetch("neuva-hub/resnet18")` — pull community models
- Version pinning and digital signatures for model integrity
- Public leaderboard for benchmark datasets (MNIST, CIFAR-10, Iris)
