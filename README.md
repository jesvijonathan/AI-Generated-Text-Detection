# AI-Generated-Text-Detection

This is gonna be revolutionary, will update soon :)

## About

An AI text/content detection model that uses GPT-4, DebertaModel and other techniques to detect & score the content/text . Use it at your own risk.

## Usage

```bash
# clone the repository
git clone

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
```

## Note

- This project works best with text with more than 250 characters & is trained on lengthy texts & using bert/DebertaModel along with 3M GPT-4 Data.
- Use this project along with the frontend interface intended to use with this project.
- Adjust the params in the config.py file as per best need
- The model is trained on a custom dataset and results are inaccurate & is not perfect, use it at your own risk.

## License

- [MIT](https://choosealicense.com/licenses/mit/)

## Contributors

- [Jesvi Jonathan](jesvi22j@gmail.com)
