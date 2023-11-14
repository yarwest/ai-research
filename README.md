# ai-research
Research into video/audio/image diffusion techniques

## Current functionalities

1. Coqui-TTS: text2voice implementation including voice cloning based on sample data. Currently working on training model on custom dataset. Includes dataset analysis script.
2. HuggingFace Stable Diffusion: text2img, img2img, inpainting, image variation. Working on video2video as well as prompt automation.
3. Whisper AI: audio transcription, created solution to generate labeled dataset for use in training TTS models. Speaker diarization to cluster detected segments by number of speakers.
4. PyTorch Object Detection: training of object detection model as well as performing object detection on images.

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

## HuggingFace Stable Diffusion (hf_stablediffusion)

### text2img

Generate an image based on a text prompt.

`$ python3 image.py --prompt "alien" --img_h 768 --num_inference_steps 25 --seed 1024`

### img2img

Transform an existing image based on a text prompt.

`$ python3 image.py --prompt "4k, dslr, sharp focus, high resolution, realistic, beautiful" --base_img "out\edits\asia grocery store.png" --img_h 768 --num_inference_steps 10 --seed 1024`

Required params: base_img

### inpainting

White pixels in the mask are repainted based on the prompt, while black pixels are preserved.

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

### Considerations

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

### Sources:

1.1. Stable Diffusion with 🧨 Diffusers - Published August 22, 2022: https://huggingface.co/blog/stable_diffusion

1.2. Stable Diffusion 1 vs 2 - What you need to know - Published December 6, 2022: https://www.assemblyai.com/blog/stable-diffusion-1-vs-2-what-you-need-to-know/

1.3. Text-to-Video: The Task, Challenges and the Current State - Published May 8, 2023: https://huggingface.co/blog/text-to-video

1.4. Prompt guide: https://stable-diffusion-art.com/prompt-guide

1.5. 10 prompts for realistic photography portraits: https://medium.com/mlearning-ai/10-ai-prompts-for-realistic-photography-portraits-da5edeacb031

1.6. Inpainting, Preserving the Unmasked Area of the Image: https://huggingface.co/docs/diffusers/using-diffusers/inpaint#preserving-the-unmasked-area-of-the-image

1.7. Evaluating diffusion models: https://huggingface.co/docs/diffusers/conceptual/evaluation

1.8. https://publicprompts.art/

1.9. Prompts Dataset: https://huggingface.co/datasets/Gustavosta/Stable-Diffusion-Prompts

1.10. Bad hands text embedding: https://huggingface.co/yesyeahvh/bad-hands-5

1.11. LoRA: https://www.ml6.eu/blogpost/low-rank-adaptation-a-technical-deep-dive

## Coqui TTS (coqui_tts)

### Voice cloning

`$ python3 voice.py --prompt $'Hello, how are you? I am good, it is a nice day out. A little bit cold, yeah. It sucks, very, so ever correct, sir!\nAre you going to have some dinnner tonight? What will you eat?\nI hope you will enjoy your meal, just like I will.\nHave a good evening! See you later.' --voice "voice_file"`

### Analyze dataset

See sources 2.3

`$ python3 analyzeDataset.py --dataset_dir "data/yarnoDataset"`

### Train model on custom dataset

`$ CUDA_VISIBLE_DEVICES=0 python3 train.py`

### TTS using model trained on custom dataset

`$ python3 voiceYarno.py --prompt "Hello everyone, I am Yarno. Today I will talk to you about generative AI and how Stable Diffusion technology can be used for social media content creation"`

### Sources:

2.1. Coqui TTS: https://github.com/coqui-ai/tts

2.2. Formatting TTS Dataset: https://tts.readthedocs.io/en/latest/formatting_your_dataset.html

2.3. Analyze Dataset Notebook: https://github.com/coqui-ai/TTS/blob/dev/notebooks/dataset_analysis/AnalyzeDataset.ipynb

2.4. Train YourTTS Notebook: https://github.com/coqui-ai/TTS/blob/dev/recipes/vctk/yourtts/train_yourtts.py

2.5. Speech Synthesis papers: https://paperswithcode.com/task/speech-synthesis/

## Audio Transcription (Whisper AI)

By default, script reads from `data/` folder and performs transcription of every audio file it finds.
The raw transcription is stored to prevent transcribing repeatedly. The transcribed segments are then used to slice the original audio file into smaller files, along with a `metadata.txt` which contains the transcription along with each file.
Path parameter is optional.

### Transcribe

`$ python3 transcribe.py --path "path/to/input.wav"`

On Windows, if you encounter `FileNotFoundError: [WinError 2] The system cannot find the file specified`, download ffmpeg.exe at: https://www.videohelp.com/software/ffmpeg

### Split audio

Provide the `--split True` flag to slice the audio file into multiple seperate .wav files.
Created for the purpose of creating labeled TTS data. Transcriber returns seperate segments, which are bundled into sentences (see `sliceSegments` & `isCompleteScentence` in `transcribe.py`) and exported to a directory in the output folder.

`$ python3 transcribe.py --path "path/to/input.wav" --split True`

The output folder created by this process can be copied to the `coqui_tts` implementation where the data set can be analyzed and loaded to train a TTS model.
This output folder also includes a `metadata.txt` containing all the transcriptions for the matching audio splits.

### Transcribe with speaker detection

Using pyannote.audio speaker diarization technique to create speaker embedings

`$ python3 transcribe.py --path "path/to/input.wav" --speakers 2`

### Sources:

3.1. https://github.com/openai/whisper

https://github.com/openai/whisper/discussions/264

3.2. Speaker Diarization implementation (MONO CHANNEL!): https://colab.research.google.com/drive/11ccdRYZSHBbUYI9grn6S1O67FVyvCKyL

3.3. Speaker Diarization resources: https://huggingface.co/spaces/openai/whisper/discussions/4

3.4. Speaker Diarization Cluster Plot: https://medium.com/@xriteshsharmax/speaker-diarization-using-whisper-asr-and-pyannote-f0141c85d59a

## Image Object Detection (PyTorch)

### Getting started

Comment line 169 (`LayerId = cv2.dnn.DictValue`) in `pytorch_objectDetection/.venv/lib/python3.10/site-packages/cv2/typing/__init__.py`

`$ python3 train.py`

`$ python3 predict.py -m version_49/checkpoints/epoch=0-step=715.ckpt -i "AIsia store.png"`

### Sources:

4.1. VOC Object Detection Dataset: https://pjreddie.com/projects/pascal-voc-dataset-mirror/

4.2. Foreground extraction using grabcut: https://www.geeksforgeeks.org/python-foreground-extraction-in-an-image-using-grabcut-algorithm/