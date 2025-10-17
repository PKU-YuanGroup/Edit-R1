# Reproduction

## GEdit-Bench

Qwen Image Edit [2509] Baseline:

```
python reproduction/sampling/sampling_qwen_gedit.py \
    --output_dir [Absolute_Output_Path] \
    --seed 42
```

UniWorld-R1-Qwen-Image-Edit [2509]:

```
python reproduction/sampling/sampling_qwen_gedit.py \
    --output_dir [Absolute_Output_Path] \
    --seed 42 \
    --lora_path [Our_LoRA]
```

> Refer to official evaluation code in GEdit. We highly recommend that set temperature=0.0 before evaluation.

## ImgEdit

Qwen Image Edit [2509] Baseline:

```
python reproduction/sampling/sampling_qwen_gedit.py \
    --output_dir [Absolute_Output_Path] \
    --seed 42
```

UniWorld-R1-Qwen-Image-Edit [2509]:

```
python reproduction/sampling/sampling_qwen_gedit.py \
    --output_dir [Absolute_Output_Path] \
    --seed 42 \
    --lora_path [Our_LoRA]
```

> Refer to official evaluation code in GEdit. We highly recommend that set temperature=0.0 before evaluation.