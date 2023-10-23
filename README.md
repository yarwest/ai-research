# ai-research
Research into video/image diffusion techniques

## Getting started

It is a good practice to logically seperate your Python projects through the use of virtual environments, this ensures that only the packages that are explicitely installed in this specific environment will be available. To manually create a virtual environment on MacOS and Linux you can use the following command line command:

`$ python3 -m venv .venv`

After the initialization process completes and the virtualenv is created, you can use the following command to activate your virtualenv and start using it:

`$ source .venv/bin/activate`

If you are using a Windows platform, you need to activate the virtualenv like this:

`% .venv\Scripts\activate.bat`

After activating the virtual environment, the required dependencies (see `requirements.txt`) need to be installed, this can be done using pip:

`$ pip3 install -r requirements.txt`

After succesfully fulfilling the dependencies, the model inference can simply be executed by running the relevant script.

### text2img

`$ python3 image.py --prompt "alien" --img_h 768 --num_inference_steps 25 --seed 1024`

### img2img

`$ python3 image.py --prompt "4k, dslr, sharp focus, high resolution, realistic, beautiful" --base_img "out\edits\asia grocery store.png" --img_h 768 --num_inference_steps 10 --seed 1024`

Required params: base_img

### inpainting

Required params: base_img, mask_img

`$ python3 image.py --prompt "shop keeper and customers, photo, photography, dslr, humans, indoor lighting, national geographic, sharp focus, high resolution, realistic, beautiful, 4k" --base_img "out\edits\asia grocery store.png" --mask_img "out/edits/AIsia store bw.png" --img_h 768 --num_inference_steps 25 --seed 1024`

#### Strict mask

"Generally speaking, StableDiffusionInpaintPipeline (and other inpainting pipelines) will change the unmasked part of the image as well. If this behavior is undesirable, you can force the unmasked area to remain the same." (source 1.6)

Strict mask implementation for preserving all pixels from base image based on black pixels in mask image.

Required params: base_img, mask_img, strict_mask

`$ python image.py --prompt "face, fingers, portrait, photo, photography, dslr, humans, highly detailed, national geographic, sharp focus, realistic, beautiful, 4k" --base_img "out\shop_keeper_and_customers,_photo,_photography,_dslr,_humans,_indoor_lighting,_national_geographic,_sharp_focus,_realistic,_beautiful,_4k-inpainting-1024-50its-0.png" --mask_img "out/edits/AIsia gen bw.png" --strict_mask True --img_h 768 --num_inference_steps 15 --seed 1024`

### variation

Required params: base_img, variation

`$ python3 image.py --prompt "alien" --base_img "path/to/base.png" --variation True --img_h 768 --num_inference_steps 25 --seed 1024`

## Considerations

Hyperparameters parameters

- Seed: determines starting point of diffusion process. Repeating the process with the same seed will lead to the same output
- num_inference_steps: number of itterations in diffusion, more itterations leads to a sharper image but will take longer
- negative_prompts: words to be excluded in diffusion target

By default, stable diffusion produces images of 512 × 512 pixels. It's very easy to override the default using the height and width arguments to create rectangular images in portrait or landscape ratios.

When choosing image sizes, we advise the following:

- Make sure height and width are both multiples of 8.
- Going below 512 might result in lower quality images.
- Going over 512 in both directions will repeat image areas (global coherence is lost).
- The best way to create non-square images is to use 512 in one dimension, and a value larger than that in the other one.

## Sources:

1.1. Stable Diffusion with 🧨 Diffusers - Published August 22, 2022: https://huggingface.co/blog/stable_diffusion
1.2. Stable Diffusion 1 vs 2 - What you need to know - Published December 6, 2022: https://www.assemblyai.com/blog/stable-diffusion-1-vs-2-what-you-need-to-know/
1.3. Text-to-Video: The Task, Challenges and the Current State - Published May 8, 2023: https://huggingface.co/blog/text-to-video
1.4. Prompt guide: https://stable-diffusion-art.com/prompt-guide
1.5. 10 prompts for realistic photography portraits: https://medium.com/mlearning-ai/10-ai-prompts-for-realistic-photography-portraits-da5edeacb031
1.6. Inpainting, Preserving the Unmasked Area of the Image: https://huggingface.co/docs/diffusers/using-diffusers/inpaint#preserving-the-unmasked-area-of-the-image

2.1. Speech Synthesis papers: https://paperswithcode.com/task/speech-synthesis/