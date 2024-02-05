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

    def handle_input_files(self, image_1, image_2, controlnet_image):
        image_1_filename = f"left{os.path.splitext(image_1)[1]}"
        image_2_filename = f"right{os.path.splitext(image_2)[1]}"
        shutil.copy(image_1, os.path.join(INPUT_DIR, image_1_filename))
        shutil.copy(image_2, os.path.join(INPUT_DIR, image_2_filename))

        if controlnet_image:
            controlnet_filename = f"controlnet{os.path.splitext(controlnet_image)[1]}"
            shutil.copy(controlnet_image, os.path.join(INPUT_DIR, controlnet_filename))
            return image_1_filename, image_2_filename, controlnet_filename

        return image_1_filename, image_2_filename, None

    def update_workflow(
        self,
        workflow,
        image_1_filename,
        image_1_weight,
        image_2_filename,
        image_2_weight,
        width,
        height,
        steps,
        control_image,
        prompt,
        negative_prompt,
        seed,
        is_upscale,
        upscale_steps,
        merge_mode,
    ):
        efficient_loader = workflow["4"]["inputs"]
        upscaler = workflow["71"]["inputs"]
        sampler = workflow["8"]["inputs"]
        ip_adapter_1 = workflow["62"]["inputs"]
        ip_adapter_2 = workflow["61"]["inputs"]

        ip_adapter_1["weight"] = image_1_weight
        ip_adapter_2["weight"] = image_2_weight

        efficient_loader["positive"] = prompt
        efficient_loader["negative"] = negative_prompt
        efficient_loader["empty_latent_width"] = width
        efficient_loader["empty_latent_height"] = height

        sampler["seed"] = seed
        sampler["steps"] = steps

        if merge_mode == "full":
            del ip_adapter_1["attn_mask"]
            del ip_adapter_2["attn_mask"]
        else:
            workflow["56"]["inputs"]["width"] = width
            workflow["56"]["inputs"]["height"] = height
            workflow["58"]["inputs"]["width"] = width
            workflow["58"]["inputs"]["height"] = height

            if merge_mode == "top_bottom":
                workflow["57"]["inputs"]["top"] = height // 2
                offset = height // 4
            elif merge_mode == "left_right":
                workflow["57"]["inputs"]["left"] = width // 2
                offset = width // 4

            self.set_mask_offset(workflow, merge_mode, offset)

        if not control_image:
            del efficient_loader["cnet_stack"]

            # Hack to stop erroring on missing file
            workflow["20"]["inputs"]["image"] = image_1_filename
        else:
            workflow["20"]["inputs"]["image"] = control_image

        workflow["10"]["inputs"]["image"] = image_1_filename
        workflow["25"]["inputs"]["image"] = image_2_filename

        if is_upscale:
            upscaler["seed"] = seed
            upscaler["steps"] = upscale_steps
            workflow["9"]["class_type"] = "PreviewImage"
        else:
            del workflow["72"]
            del upscaler["image"]
            del upscaler["model"]
            del upscaler["positive"]
            del upscaler["negative"]
            del upscaler["vae"]

    def set_mask_offset(self, workflow, merge_mode, offset):
        if merge_mode == "left_right":
            workflow["59"]["inputs"]["x"] = offset
        elif merge_mode == "top_bottom":
            workflow["59"]["inputs"]["y"] = offset

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
        image_1_strength: float = Input(
            ge=0, le=1, default=1, description="The strength of the first image"
        ),
        image_2: Path = Input(),
        image_2_strength: float = Input(
            ge=0, le=1, default=1, description="The strength of the second image"
        ),
        merge_mode: str = Input(
            default="full",
            choices=["full", "left_right", "top_bottom"],
            description="The mode to use for merging the images",
        ),
        prompt: str = Input(
            default="a photo", description="A prompt to guide the image merging"
        ),
        negative_prompt: str = Input(
            default="ugly, broken, distorted",
            description="Things you do not want in the merged image",
        ),
        width: int = Input(default=768),
        height: int = Input(default=768),
        steps: int = Input(default=20),
        control_image: Path = Input(
            default=None,
            description="An optional image to use with control net to influence the merging",
        ),
        seed: int = Input(
            default=None, description="Fix the random seed for reproducibility"
        ),
        upscale_2x: bool = Input(default=False),
        upscale_steps: int = Input(
            default=20, description="The number of steps per controlnet tile"
        ),
        animate: bool = Input(
            default=False,
            description="Animate merging from one image to the other. Only the video is returned.",
        ),
        animate_frames: int = Input(
            default=24, description="The number of frames to generate for the animation"
        ),
        return_temp_files: bool = Input(
            description="Return any temporary files, such as preprocessed controlnet images. Useful for debugging.",
            default=False,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.cleanup()

        if not image_1 or not image_2:
            raise ValueError("Please provide two input images")

        if animate and (merge_mode == "full" or not control_image):
            raise ValueError(
                "Animation is only supported for left_right and top_bottom merge modes with a control image"
            )

        (
            image_1_filename,
            image_2_filename,
            controlnet_filename,
        ) = self.handle_input_files(image_1, image_2, control_image)

        if seed is None:
            seed = random.randint(0, 2**32 - 1)
            print(f"Random seed set to: {seed}")

        workflow = json.loads(workflow_json)

        self.update_workflow(
            workflow,
            image_1_filename,
            image_1_strength,
            image_2_filename,
            image_2_strength,
            width,
            height,
            steps,
            controlnet_filename,
            prompt,
            negative_prompt,
            seed,
            upscale_2x,
            upscale_steps,
            merge_mode,
        )

        wf = self.comfyUI.load_workflow(workflow)
        self.comfyUI.connect()

        if animate:
            dimension = width if merge_mode == "left_right" else height
            step_size = max(
                1,
                dimension // animate_frames,
            )
            print(f"Dimension: {dimension}")
            print(f"Step size: {step_size}")
            for frame_number in range(animate_frames):
                offset = max(1, step_size * frame_number)
                print(f"Running frame {frame_number + 1} of {animate_frames}")
                print(f"Offset: {offset}")
                self.set_mask_offset(wf, merge_mode, offset)
                self.comfyUI.run_workflow(wf)
        else:
            self.comfyUI.run_workflow(wf)

        files = []
        output_directories = [OUTPUT_DIR]

        if return_temp_files:
            output_directories.append(COMFYUI_TEMP_OUTPUT_DIR)

        for directory in output_directories:
            print(f"Contents of {directory}:")
            files.extend(self.log_and_collect_files(directory))

        if animate:
            video_output_filename = os.path.join(OUTPUT_DIR, "output_video.mp4")
            ffmpeg_command = [
                "ffmpeg",
                "-r",
                "12",
                "-pattern_type",
                "glob",
                "-i",
                f"{OUTPUT_DIR}/*.png",  # Use file order and only PNG images
                "-c:v",
                "libx264",  # Video codec to be used
                "-pix_fmt",
                "yuv420p",  # Pixel format for compatibility
                "-vf",
                "format=yuv420p",  # Set the video format to yuv420p
                "-y",  # Overwrite output file if it exists
                video_output_filename,  # Output filename
            ]
            try:
                subprocess.run(ffmpeg_command, check=True)
                print(f"Video successfully created at {video_output_filename}")
            except subprocess.CalledProcessError as e:
                print(f"An error occurred while creating the video: {e}")

            return [Path(video_output_filename)]

        return files
