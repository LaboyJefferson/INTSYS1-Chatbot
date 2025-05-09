# CHATBOT SYSTEM README

## Prerequisites

### Python:
Python 3.11.1 or higher

### Imports:
The system requires the following Python libraries. To install them, you can use the `requirements.txt` file provided.

Install the following:
*open the terminal of this pycharm project on the sidebar or press ALT+F12
    > pip install -r requirements.txt
    > python -m spacy download en_core_web_sm

- Flask == 3.1.0
- chatterbot == 1.2.6
- torch == 2.7.0
- nltk == 3.9.1
- numpy == 2.2.5

### How to run

1. Run the 'train.py' script
    This will train the model first with the training data under the train_data directory.

    Run the following command:
    > python train.py

2. Run the 'server.py' script
    You can start the server to make the chatbot accessible via a web interface.

    Run the following command:
    > python server.py

    The server will run locally.