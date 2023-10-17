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