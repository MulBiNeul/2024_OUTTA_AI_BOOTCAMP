import os, sys
import argparse
import copy
from IPython.display import display
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert
import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL
import requests
import torch
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline
from huggingface_hub import hf_hub_download

def preprop_for_diffusion(image, vis_output_model):

    image_t = image.transpose(2, 0, 1)
    array_transposed1 = np.transpose(image_t, (1, 2, 0))

    image1 = np.rot90(array_transposed1, k=3)
    # plt.imshow(image1)  # 변환된 이미지를 시각적으로 확인하기 위한 코드
    # plt.show()

    array_transposed2 = vis_output_model

    mask_image1 = np.rot90(array_transposed2, k=3)
    # plt.imshow(mask_image1)  # 변환된 마스크 이미지를 시각적으로 확인하기 위한 코드
    # plt.show()

    image1 = image1 * 256
    image1 = image1.astype(np.uint8)
    mask_image1 = mask_image1.astype(np.uint8)

    image_source_pil = Image.fromarray(image1)
    image_mask_pil = Image.fromarray(mask_image1)

    display(*[image_source_pil, image_mask_pil])

    return image_source_pil, image_mask_pil


def generate_image(image, mask, prompt, negative_prompt, pipe, seed, device):
    w, h = image.size
    in_image = image.resize((512, 512))
    in_mask = mask.resize((512, 512))

    generator = torch.Generator(device).manual_seed(seed)

    result = pipe(prompt=prompt, negative_prompt=negative_prompt, image=in_image, mask_image=in_mask, generator=generator).images

    result = result[0]

    return result.resize((w, h))