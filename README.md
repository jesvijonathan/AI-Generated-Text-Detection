# AI-Generated-Text-Detection

This is gonna be revolutionary, will update soon :)

## About

An AI text/content detection model that uses GPT-4, DebertaModel and other techniques to detect & score the content/text. [Snitch-GPT](https://snitch-gpt.vercel.app) ([source](https://github.com/jesvijonathan/Snitch-GPT-Frontend))

## Usage

```bash
# clone the repository
git clone

# Download weights (~6GB)
https://www.mediafire.com/file/byct4m37gdvs36n/weights.7z/file
# extract and place under weights folder

# navigate to the project directory
cd AI-Generated-Text-Detection

# create a python environment | python version 3.10.6
python -m venv env

# activate the environment
source env/bin/activate

# install the required packages
pip install -r packages.txt
# you require gpu/cuda version of tensorflow/torch and cuda drivers to run the model

# run the main.py file
python app.py

# use the app via API or web interface
# https://github.com/jesvijonathan/Snitch-GPT-Frontend
```

## Note

- This project works best with text with more than 250 characters & is trained on lengthy texts & using bert/DebertaModel along with 3M GPT-4 Data.
- Use this project along with the frontend interface intended to use with this project, which has visualization & more | [Snitch-GPT](https://snitch-gpt.vercel.app)
- Adjust the params in the config.py file best, as per need
- The model is trained on a custom dataset and results are inaccurate & is not perfect, use it at your own risk.
- I know the code is speghetti but it works, will refactor soon.

## License

- [MIT](https://choosealicense.com/licenses/mit/)

## Contributors

- [Jesvi Jonathan](jesvi22j@gmail.com)
