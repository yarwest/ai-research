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

`$ python3 train.py`

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

Stable Diffusion with 🧨 Diffusers - Published August 22, 2022: https://huggingface.co/blog/stable_diffusion
Stable Diffusion 1 vs 2 - What you need to know - Published December 6, 2022: https://www.assemblyai.com/blog/stable-diffusion-1-vs-2-what-you-need-to-know/
