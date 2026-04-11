These `.pt` files are placeholder checkpoints for notebook experiments.

For now they contain JSON metadata so the comparison notebook can run without real training weights.

Later you can either:

1. Replace the file contents with your real PyTorch checkpoint while keeping the same filename, or
2. Point the notebook config to a different checkpoint path.

Supported real-checkpoint architectures in the notebook helper layer are:

- `resnet18`
- `vit_b_16`
- `vgg16`
