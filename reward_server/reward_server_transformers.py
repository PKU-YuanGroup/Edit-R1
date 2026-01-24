import os
import torch
from typing import List
from PIL import Image
from io import BytesIO
import pickle
import traceback
from flask import Flask, request
import ray
import asyncio
import prompt_template

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

app = Flask(__name__)

# Global variables
score_idx = [15, 16, 17, 18, 19, 20]  # Token IDs for scores 0-5
workers = []
MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
NUM_GPUS = 2


@ray.remote(num_gpus=1)
class ModelWorker:
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self.model = None
        self.processor = None
        self.load_model()

    def load_model(self):
        """Load the Qwen2-VL model using transformers on specific GPU"""
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
        ).cuda()
        self.processor = AutoProcessor.from_pretrained(MODEL_PATH)
        self.model.eval()

    def evaluate_image(
        self, image_bytes, prompt, ref_image_bytes=None, requirement: str = ""
    ):
        """Evaluate image pair and return score"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(BytesIO(image_bytes), formats=["jpeg"])
            ref_image = Image.open(BytesIO(ref_image_bytes), formats=["jpeg"])

            # Build conversation
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": ref_image},
                        {"type": "image", "image": image},
                        {
                            "type": "text",
                            "text": prompt_template.SCORE_LOGIT.format(
                                prompt=prompt, requirement=requirement
                            ),
                        },
                    ],
                },
            ]

            return self._transformers_evaluate(messages)
        except Exception as e:
            print(f"Error in evaluate_image: {e}")
            traceback.print_exc()
            return 0.0

    def _transformers_evaluate(self, messages, max_tokens=3, max_score=5):
        """Evaluate using transformers and extract score from logits"""
        try:
            # Process the conversation
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Generate with output scores
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    return_dict_in_generate=True,
                    output_logits=True,
                )

            # Get the first token logits
            if outputs.logits:
                first_token_logits = outputs.logits[0][-1, score_idx]  # [vocab_size]
                probs = torch.softmax(first_token_logits, dim=-1)
                print(probs, max_score)
                score_prob = (
                    torch.sum(
                        probs * torch.arange(len(score_idx)).to(probs.device)
                    ).item()
                    / max_score
                )
                print(f"Score: {score_prob:.4f}")
                return score_prob
            else:
                print("No outputs received")
                return 0.0

        except Exception as e:
            print(f"Error in _transformers_evaluate: {e}")
            traceback.print_exc()
            return 0.0


def initialize_ray_workers(num_gpus=2):
    global workers
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()

    available_gpus = torch.cuda.device_count()
    if available_gpus <= 0:
        raise RuntimeError("No CUDA devices available")
    print("Available GPUs: ", available_gpus, "num_gpus: ", num_gpus)
    num_gpus = min(num_gpus, available_gpus)

    # Create workers for each GPU
    workers = []
    for gpu_id in range(num_gpus):
        worker = ModelWorker.remote(gpu_id)
        workers.append(worker)

    print(f"Initialized {num_gpus} Ray workers")
    return workers


async def evaluate_images_async(
    image_bytes_list, prompts, ref_image_bytes_list=None, requirements: List[str] = []
):
    global workers

    if not workers:
        raise RuntimeError("Ray workers not initialized")

    tasks = []
    if not requirements:
        requirements = [""] * len(prompts)
    if ref_image_bytes_list is None:
        ref_image_bytes_list = [None] * len(prompts)

    for i, (image_bytes, prompt, ref_image_bytes, requirement) in enumerate(
        zip(image_bytes_list, prompts, ref_image_bytes_list, requirements)
    ):
        worker_idx = i % len(workers)
        worker = workers[worker_idx]
        task = worker.evaluate_image.remote(
            image_bytes, prompt, ref_image_bytes, requirement
        )
        tasks.append(task)

    scores = ray.get(tasks)
    return scores


def evaluate_images(
    image_bytes_list, prompts, ref_image_bytes_list=None, requirements=[]
):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        scores = loop.run_until_complete(
            evaluate_images_async(
                image_bytes_list, prompts, ref_image_bytes_list, requirements
            )
        )
        return scores
    finally:
        loop.close()


@app.route("/mode/<mode>", methods=["POST"])
def inference_mode(mode):
    data = request.get_data()

    assert mode in ["logits_non_cot"], "Invalid mode"

    try:
        data = pickle.loads(data)
        image_bytes_list = data["images"]
        ref_image_bytes_list = data.get("ref_images", None)
        prompts = data["prompts"]
        metadatas = data.get("metadatas", [])
        requirements = []
        for metadata in metadatas:
            requirements.append(metadata.get("requirement", ""))

        scores = evaluate_images(
            image_bytes_list, prompts, ref_image_bytes_list, requirements
        )

        response = {"scores": scores}
        response = pickle.dumps(response)
        returncode = 200
    except KeyError as e:
        response = f"KeyError: {str(e)}"
        response = response.encode("utf-8")
        returncode = 500
    except Exception as e:
        response = traceback.format_exc()
        response = response.encode("utf-8")
        returncode = 500

    return response, returncode


if __name__ == "__main__":
    initialize_ray_workers(NUM_GPUS)
    print(f"Starting Flask server with {NUM_GPUS} Ray workers...")
    app.run(host="0.0.0.0", port=12341, debug=False)

