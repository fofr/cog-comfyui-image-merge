import os
import shutil
import json
import random
import subprocess
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
        self.comfyUI.load_workflow(workflow_json, handle_inputs=False)

    def cleanup(self):
        for directory in [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)

    def handle_input_files(self, image_1, image_2):
        image_1_filename = f"left{os.path.splitext(image_1)[1]}"
        image_2_filename = f"right{os.path.splitext(image_2)[1]}"
        shutil.copy(image_1, os.path.join(INPUT_DIR, image_1_filename))
        shutil.copy(image_2, os.path.join(INPUT_DIR, image_2_filename))
        return image_1_filename, image_2_filename

    def update_workflow(
        self,
        workflow,
        base_model,
        image_1_filename,
        image_2_filename,
        merge_strength,
        width,
        height,
        steps,
        prompt,
        negative_prompt,
        seed,
        batch_size=1,
        noise=0.8,
    ):
        loader = workflow["4"]["inputs"]
        first_image = workflow["7"]["inputs"]
        second_image = workflow["16"]["inputs"]
        sampler = workflow["8"]["inputs"]
        ip_adapter = workflow["15"]["inputs"]

        loader["ckpt_name"] = base_model
        loader["positive"] = prompt
        loader["negative"] = negative_prompt
        loader["empty_latent_width"] = width
        loader["empty_latent_height"] = height
        loader["batch_size"] = batch_size
        ip_adapter["weight"] = merge_strength
        ip_adapter["noise"] = noise
        first_image["image"] = image_1_filename
        second_image["image"] = image_2_filename
        sampler["steps"] = steps
        sampler["seed"] = seed

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
        base_model: str = Input(
            default="albedobaseXL_v13.safetensors",
            choices=[
                "albedobaseXL_v13.safetensors",
                "juggernautXL_v8Rundiffusion.safetensors",
                "proteus_v02.safetensors",
                "RealVisXL_V3.0.safetensors",
                "RealVisXL_V4.0.safetensors",
                "sd_xl_base_1.0.safetensors",
                "starlightXLAnimated_v3.safetensors",
            ],
            description="Select the base model for the prediction",
        ),
        image_1: Path = Input(),
        image_2: Path = Input(),
        merge_strength: float = Input(
            ge=0,
            le=1,
            default=1,
            description="Reduce strength to increase prompt weight",
        ),
        added_merge_noise: float = Input(
            ge=0.0,
            le=1.0,
            default=0.8,
            description="More noise allows for more prompt control",
        ),
        prompt: str = Input(
            default="a photo", description="A prompt to guide the image merging"
        ),
        negative_prompt: str = Input(
            default="",
            description="Things you do not want in the merged image",
        ),
        width: int = Input(default=1024),
        height: int = Input(default=1024),
        steps: int = Input(default=20),
        seed: int = Input(
            default=None, description="Fix the random seed for reproducibility"
        ),
        batch_size: int = Input(
            ge=1, le=8, default=1, description="The batch size for the model"
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.cleanup()

        if not image_1 or not image_2:
            raise ValueError("Please provide two input images")

        (
            image_1_filename,
            image_2_filename,
        ) = self.handle_input_files(image_1, image_2)

        if seed is None:
            seed = random.randint(0, 2**32 - 1)
            print(f"Random seed set to: {seed}")

        workflow = json.loads(workflow_json)

        self.update_workflow(
            workflow,
            base_model,
            image_1_filename,
            image_2_filename,
            merge_strength,
            width,
            height,
            steps,
            prompt,
            negative_prompt,
            seed,
            batch_size,
            noise=added_merge_noise,
        )

        wf = self.comfyUI.load_workflow(workflow)
        self.comfyUI.connect()

        self.comfyUI.run_workflow(wf)

        files = []
        output_directories = [OUTPUT_DIR]

        for directory in output_directories:
            print(f"Contents of {directory}:")
            files.extend(self.log_and_collect_files(directory))

        return files
