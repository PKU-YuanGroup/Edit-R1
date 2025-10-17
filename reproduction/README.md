# Reproduction

## GEdit-Bench

Qwen Image Edit [2509] Baseline:

```
python reproduction/sampling/sampling_qwen_gedit.py \
    --pretrained_name_or_path [pretrained_model] \
    --gedit_bench_path [gedit_bench_path] \
    --output_dir [absolute_output_path] \
    --seed 42
```

UniWorld-Qwen-Image-Edit [2509]:

```
python reproduction/sampling/sampling_qwen_gedit.py \
    --pretrained_name_or_path [pretrained_model] \
    --gedit_bench_path [gedit_bench_path] \
    --output_dir [absolute_output_path] \
    --seed 42 \
    --lora_path [our_lora]
```

> Refer to official evaluation code in GEdit. We highly recommend that set `temperature=0.0` before evaluation.

## ImgEdit

Qwen Image Edit [2509] Baseline:

```
python reproduction/sampling/sampling_qwen_imgedit.py \
    --pretrained_name_or_path [pretrained_model] \
    --output_dir [absolute_output_path] \
    --seed 42
```

UniWorld-Qwen-Image-Edit [2509]:

```
python reproduction/sampling/sampling_qwen_imgedit.py \
    --pretrained_name_or_path [pretrained_model] \
    --input_path "[singleturn_json]" \
    --output_dir "[absolute_output_path]" \
    --root_path "[singleturn_dir]" \
    --seed 42 \
    --lora_path [our_lora]
```

> Refer to official evaluation code in ImgEdit. We highly recommend that set `temperature=0.0` before evaluation.
