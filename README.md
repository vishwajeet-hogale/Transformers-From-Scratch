# Transformers From Scratch

PyTorch implementations of two vision transformers, written by hand to understand how they work:

- **Vision Transformer (ViT)** for image representation
- **DETR (DEtection TRansformer)** for object detection, trained on the CPPE-5 dataset

Each model lives in its own notebook with the code, shape checks, and outputs.

## Notebooks

| Notebook | What it covers |
| --- | --- |
| [`vit_from_scratch.ipynb`](vit_from_scratch.ipynb) | Patch embedding, sinusoidal positional encoding, multi-head self-attention, pre-norm transformer blocks, CLS token |
| [`detr_from_scratch.ipynb`](detr_from_scratch.ipynb) | IoU / GIoU / DIoU, dataset pipeline, object queries, Hungarian matching, set-based loss, training with match visualizations |

## Vision Transformer

Follows [Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929).

```
Image (B, 3, 256, 256)
  -> 16x16 patch embedding        (B, 256, 768)
  -> prepend CLS token            (B, 257, 768)
  -> add sinusoidal positions
  -> 10 transformer blocks
  -> output                       (B, 257, 768)
```

Implemented by hand:
- Patch embedding with a 16x16 convolution at stride 16
- Sinusoidal positional encoding
- Multi-head self-attention: Q, K, V projections, head splitting, scaled dot-product attention, optional masking, output projection
- Pre-norm transformer block with a GELU feed-forward network and residual connections
- Learnable CLS token

## DETR

A simplified version of [Carion et al., 2020](https://arxiv.org/abs/2005.12872), trained on [CPPE-5](https://huggingface.co/datasets/rishitdagli/cppe-5), a medical PPE dataset with 5 classes: coverall, face shield, gloves, goggles, and mask.

```
Image (B, 3, 256, 256)
  -> 16x16 patch tokens + learnable positions   (B, 256, 768)
  -> transformer encoder (3 layers)
  -> transformer decoder (3 layers) with 200 learnable object queries
  -> class head   (B, 200, 6)   5 classes + "no object"
  -> box head     (B, 200, 4)   normalized cx, cy, w, h
```

Implemented by hand:
- **Box metrics**: IoU, Generalized IoU, and Distance IoU, with unit tests
- **Data pipeline**: COCO `xywh` boxes converted to normalized `cxcywh`, plus a collate function for images with different numbers of objects
- **Hungarian matching**: one-to-one assignment of predictions to ground truth using a combined class, L1, and GIoU cost
- **Set loss**: cross-entropy over all queries (unmatched ones labeled "no object"), plus L1 and GIoU box losses on matched pairs
- **Visualization**: every 2 epochs, ground truth boxes are drawn next to the queries matched to them, so you can watch the matching change during training

### How this differs from the paper

- No CNN backbone. Image patches go straight into the transformer.
- The encoder-decoder uses PyTorch's `nn.Transformer` rather than a hand-written one.
- The GIoU used during training comes from `torchvision.ops`. The hand-written version in the notebook is tested but not used in the loss.
- No auxiliary decoding losses and no pretrained weights.

### Current status

Training runs end to end for 30 epochs, but the loss only drops from about 2.37 to 2.28, so the model has not learned to detect objects yet. This matches what the DETR paper reports: DETR trains slowly, especially without a pretrained backbone on a small dataset. The notebook is best read as an implementation walkthrough, not a trained detector.

## Getting started

The notebooks were built in Google Colab on a T4 GPU.

1. Open a notebook in [Colab](https://colab.research.google.com/) or Jupyter.
2. Install dependencies if you're running locally:
   ```bash
   pip install torch torchvision datasets scipy matplotlib
   ```
3. Run the cells in order. The DETR notebook downloads CPPE-5 from Hugging Face automatically.

A GPU is recommended for DETR training. The ViT notebook runs fine on a CPU.

## Next steps

- Replace `nn.Transformer` in DETR with the hand-written transformer blocks from the ViT notebook
- Add a pretrained CNN backbone (for example, ResNet-50) to speed up DETR convergence
- Add auxiliary decoder losses and evaluate with COCO mAP
- Train the ViT on a classification task

## References

- Dosovitskiy et al., [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929), ICLR 2021
- Carion et al., [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872), ECCV 2020
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), NeurIPS 2017
- Dagli and Shaikh, [CPPE-5: Medical Personal Protective Equipment Dataset](https://arxiv.org/abs/2112.09569), 2021
