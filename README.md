# PCSR

Official PyTorch implementation of:

> **PCSR: Pseudo-label Consistency-Guided Sample Refinement for Noisy Correspondence Learning**  
> **Authors:** Zhuoyao Liu, Yang Liu, Wentao Feng, Shudong Huang

Quick links: [[PDF](https://arxiv.org/pdf/2509.15623)] [[Abstract](https://arxiv.org/abs/2509.15623)]

## Overview
Cross-modal retrieval models trained on naturally collected image-text pairs are often affected by noisy correspondences.  
PCSR improves robustness through pseudo-label consistency-guided sample refinement in noisy correspondence learning.

## Environment
```bash
pip install -r requirements.txt
```

## Data Format
This code uses precomputed image features and caption files.

Supported datasets:
- `f30k_precomp`
- `coco_precomp`
- `cc152k_precomp`
- `now100k_precomp`

Expected directory structure (example):
```text
data/
  f30k_precomp/
    train_caps.txt
    train_ims.npy
    dev_caps.txt
    dev_ims.npy
    test_caps.txt
    test_ims.npy
  vocab/
    f30k_precomp_vocab.json
```

For `now100k_precomp`, captions should use tokenizer-specific files such as:
- `train_caps_bpe.txt`
- `train_caps_bert.txt`

and vocabulary files such as:
- `now100k_precomp_vocab_bpe.json`
- `now100k_precomp_vocab_bert.json`

## Training

Single GPU:
```bash
python run.py \
  --world-size 1 --rank 0 --gpu 0 \
  --data_path ./data \
  --data_name f30k_precomp \
  --vocab_path ./data/vocab \
  --output_dir ./output
```

Multi-GPU (single node, DDP):
```bash
python run.py \
  --world-size 1 --rank 0 \
  --multiprocessing-distributed \
  --data_path ./data \
  --data_name f30k_precomp \
  --vocab_path ./data/vocab \
  --output_dir ./output
```

Important arguments:
- `--noise_ratio`: noise ratio used for noisy-correspondence training.
- `--noise_file`: noisy index file path (if empty, generated automatically).
- `--warmup_epoch`, `--warmup_epoch_2`, `--warmup_epoch_3`: warm-up and staged training epochs.
- `--model_path`, `--po_dir`, `--resume`: options for resuming training.
- `--tokenizer`: tokenizer for `now100k_precomp` (`bpe` or `bert`).

## Evaluation
```bash
python evaluation.py \
  --data_path ./data \
  --vocab_path ./data/vocab \
  --model_path ./output/<run_dir>/checkpoint_best_validation.pth.tar \
  --gpu 0
```

## Citation
```bibtex
@article{liu2025pcsr,
  title={PCSR: Pseudo-label Consistency-Guided Sample Refinement for Noisy Correspondence Learning},
  author={Liu, Zhuoyao and Liu, Yang and Feng, Wentao and Huang, Shudong},
  journal={arXiv preprint arXiv:2509.15623},
  year={2025}
}
```

## License
This project is released under the MIT License. See `LICENSE` for details.

## Acknowledgement
This implementation builds on prior open-source cross-modal retrieval codebases, including NCR and related SGRAF-style implementations.
