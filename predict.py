import os
import shutil
import tarfile
import zipfile
import json
import random
from typing import List
from cog import BasePredictor, Input, Path
from helpers.comfyui import ComfyUI

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"

with open("workflow.json", "r") as file:
    workflow_json = file.read()


class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

    def cleanup(self):
        for directory in [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)

    def handle_input_files(self, image_1, image_2, controlnet_image):
        left_filename = f"left{os.path.splitext(image_1)[1]}"
        right_filename = f"right{os.path.splitext(image_2)[1]}"
        shutil.copy(image_1, os.path.join(INPUT_DIR, left_filename))
        shutil.copy(image_2, os.path.join(INPUT_DIR, right_filename))

        if controlnet_image:
            controlnet_filename = f"controlnet{os.path.splitext(controlnet_image)[1]}"
            shutil.copy(controlnet_image, os.path.join(INPUT_DIR, controlnet_filename))
            return left_filename, right_filename, controlnet_filename

        return left_filename, right_filename, None

    def update_workflow(
        self,
        workflow,
        left_image,
        right_image,
        width,
        height,
        control_image,
        prompt,
        negative_prompt,
        seed,
        is_upscale,
    ):
        efficient_loader = workflow["4"]["inputs"]
        upscaler = workflow["71"]["inputs"]
        sampler = workflow["8"]["inputs"]
        ip_adapter_1 = workflow["61"]["inputs"]
        ip_adapter_2 = workflow["62"]["inputs"]

        efficient_loader["positive"] = prompt
        efficient_loader["negative"] = negative_prompt
        efficient_loader["empty_latent_width"] = width
        efficient_loader["empty_latent_height"] = height

        # temporarily ignore masks
        del ip_adapter_1["attn_mask"]
        del ip_adapter_2["attn_mask"]

        sampler["seed"] = seed

        if not control_image:
            del efficient_loader["cnet_stack"]

            # Hack to stop erroring on missing file
            workflow["20"]["inputs"]["image"] = left_image
        else:
            workflow["20"]["inputs"]["image"] = control_image

        workflow["10"]["inputs"]["image"] = left_image
        workflow["25"]["inputs"]["image"] = right_image

        if is_upscale:
            upscaler["seed"] = seed
            workflow["9"]["class_type"] = "PreviewImage"
        else:
            del workflow["72"]
            del upscaler["image"]
            del upscaler["model"]
            del upscaler["positive"]
            del upscaler["negative"]
            del upscaler["vae"]

    def log_and_collect_files(self, directory, prefix=""):
        files = []
        for f in os.listdir(directory):
            if f == "__MACOSX":
                continue
            path = os.path.join(directory, f)
            if os.path.isfile(path):
                print(f"{prefix}{f}")
                files.append(Path(path))
            elif os.path.isdir(path):
                print(f"{prefix}{f}/")
                files.extend(self.log_and_collect_files(path, prefix=f"{prefix}{f}/"))
        return files

    def predict(
        self,
        image_1: Path = Input(),
        image_2: Path = Input(),
        prompt: str = Input(
            default="a photo", description="A prompt to guide the image merging"
        ),
        negative_prompt: str = Input(
            default="ugly, broken, distorted",
            description="Things you do not want in the merged image",
        ),
        width: int = Input(default=768),
        height: int = Input(default=768),
        control_image: Path = Input(
            default=None,
            description="An optional image to use with control net to influence the merging",
        ),
        seed: int = Input(
            default=None, description="Fix the random seed for reproducibility"
        ),
        upscale_2x: bool = Input(default=False),
        return_temp_files: bool = Input(
            description="Return any temporary files, such as preprocessed controlnet images. Useful for debugging.",
            default=False,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.cleanup()

        if not image_1 or not image_2:
            raise ValueError("Please provide two input images")

        left_image, right_image, controlnet_image = self.handle_input_files(
            image_1, image_2, control_image
        )

        if seed is None:
            seed = random.randint(0, 2**32 - 1)
            print(f"Random seed set to: {seed}")

        workflow = json.loads(workflow_json)

        self.update_workflow(
            workflow,
            left_image,
            right_image,
            width,
            height,
            controlnet_image,
            prompt,
            negative_prompt,
            seed,
            upscale_2x,
        )

        wf = self.comfyUI.load_workflow(workflow)

        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        files = []
        output_directories = [OUTPUT_DIR]
        if return_temp_files:
            output_directories.append(COMFYUI_TEMP_OUTPUT_DIR)

        for directory in output_directories:
            print(f"Contents of {directory}:")
            files.extend(self.log_and_collect_files(directory))

        return files
