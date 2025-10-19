<p align="center">
    <img src="https://s21.ax1x.com/2025/06/03/pVCBdw8.png" width="200"/>
<p>
<h2 align="center"> 
  <a href="https://arxiv.org/abs/2506.03147">
    UniWorld-V2: Reinforce Image Editing with Diffusion Negative-Aware Finetuning and
MLLM Implicit Feedback
  </a>
</h2>

[![UniWorld-V2](https://img.shields.io/badge/Arxiv-UniWorldV2-b31b1b.svg?logo=arXiv)]()
[![UniWorld-V1](https://img.shields.io/badge/Arxiv-UniWorldV1-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.03147)
[![ImgEdit](https://img.shields.io/badge/Arxiv-ImgEdit-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.03147)
[![Collection](https://img.shields.io/badge/🤗-Collection-blue.svg)](https://huggingface.co/collections/chestnutlzj/edit-r1-68dc3ecce74f5d37314d59f4)
[![License](https://img.shields.io/badge/License-Apache-yellow)](https://github.com/PKU-YuanGroup/UniWorld-V2/blob/main/LICENSE)

## 📣 News

**[2025/10/19]**: We release **Edit-R1**, which employs [DiffusionNFT](https://github.com/NVlabs/DiffusionNFT) and a training-free reward
model derived from pretrained MLLMs to fine-tune diffusion models for image editing. [Code](https://github.com/PKU-YuanGroup/UniWorld-V2) and [models](https://huggingface.co/collections/chestnutlzj/edit-r1-68dc3ecce74f5d37314d59f4) are open-sourced.

## 🗝️ Train

### Deploy vLLM Reward Server

Start the reward server:

```
python reward_server/reward_server.py
```

If you want to check the status of the reward server, you can test it by running:

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

## ⚡️ Reproduction

For reproducibility, we provide the reproduction scripts in `reproduction/`.

See [Reproduction Details](reproduction/README.md) for more details.

## 👍 Acknowledgement

- [**DiffusionNFT**](https://github.com/NVlabs/DiffusionNFT): Huge thanks for their elegant codebase 🤩!
- [Flow-GRPO](https://github.com/yifan123/flow_grpo)
- [ImgEdit](https://github.com/PKU-YuanGroup/ImgEdit)
- [UniWorld-V1](https://github.com/PKU-YuanGroup/UniWorld-V1)

## 🔒 License

See [LICENSE](LICENSE) for details. The FLUX weights fall under the [FLUX.1 [dev] Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md).

## ✏️ Citation

```
@article{lin2025uniworld,
  title={Uniworld: High-resolution semantic encoders for unified visual understanding and generation},
  author={Lin, Bin and Li, Zongjian and Cheng, Xinhua and Niu, Yuwei and Ye, Yang and He, Xianyi and Yuan, Shenghai and Yu, Wangbo and Wang, Shaodong and Ge, Yunyang and others},
  journal={arXiv preprint arXiv:2506.03147},
  year={2025}
}

@article{ye2025imgedit,
  title={Imgedit: A unified image editing dataset and benchmark},
  author={Ye, Yang and He, Xianyi and Li, Zongjian and Lin, Bin and Yuan, Shenghai and Yan, Zhiyuan and Hou, Bohan and Yuan, Li},
  journal={arXiv preprint arXiv:2505.20275},
  year={2025}
}
```
