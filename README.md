<p align="center">
    <img src="https://s21.ax1x.com/2025/06/03/pVCBdw8.png" width="200"/>
<p>
<h2 align="center"> 
  <a href="https://arxiv.org/abs/2506.03147">
    UniWorld-V2: Reinforce Image Editing with Diffusion Negative-Aware Finetuning and
MLLM Implicit Feedback
  </a>
</h2>
  
[![UniWorld-V1](https://img.shields.io/badge/Arxiv-2506.03147-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.03147)
[![Model & Data](https://img.shields.io/badge/🤗-Model-blue.svg)](https://huggingface.co/collections/chestnutlzj/uniworld-r1-68dc3ecce74f5d37314d59f4)
[![License](https://img.shields.io/badge/License-Apache-yellow)](https://github.com/PKU-YuanGroup/UniWorld-V2/blob/main/LICENSE)

## Train

### Deploy vLLM Reward Server

```
python reward_server/reward_server.py
```

```
python reward_server/test_reward_server.py
```

### Configure Training

See `config/qwen_image_edit_nft.py` and `config/kontext_nft.py` for available configurations.

### Run Training

```shell
export REWARD_SERVER=[YOUR_REWARD_SERVICE_IP_ADDR]:12341

torchrun --nproc_per_node=8 \
    scripts/train_nft_qwen_image_edit.py --config config/qwen_image_edit_nft.py:config_name
```

And you can also refer to the example scripts in `examples/`.

## Reproduction

For reproducibility, we provide the reproduction scripts in `reproduction/`.

See [Reproduction Details](reproduction/README.md) for more details.

## Acknowledgement

- [**DiffusionNFT**](https://github.com/NVlabs/DiffusionNFT): Huge thanks for their elegant codebase 🤩!
- [Flow-GRPO](https://github.com/yifan123/flow_grpo)
