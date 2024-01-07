# English-Italian Translation Transformer

This repository houses code to train a transformer model from scratch in PyTorch for the task of translating text from English to Italian.

## Transformer Specifications

### Model Architecture:

 - Encoder-decoder architecture with stacked attention layers
 - 6 encoder layers and 6 decoder layers
 - 6 attention heads for all Multi-Header Attention Layers
 - Positional encoding 
 - Layer normalization
 - Dropout regularization (rate = 0.1)
 - Embedding Dimension: 512
 - Sequence Length: 350
 - Dataset Used: [OPUS Books dataset](https://huggingface.co/datasets/opus_books)


## Getting Started

### Installation
 1. Clone the Repository:

        https://github.com/aap2239/english-italian-translation-transformer

 2. Create a Virtual Environment

        cd english-italian-translation-transformer
        python -m venv .
        .\venv\Scripts\activate

 3. Install all the requirements

        pip install -r requirements.txt

    You might have to install CUDA on your local machine to run the model on your GPU. 

### Training of the Transformer

 1. Train the Model using

        python train_en_it.py

 2. While training you can also view the loss on ``tensorboard`` running the following command on a new terminal and going to the link provided:

        tensorboard --logdir=runs



### Inference / Translation using the model 

 1. Once you have trained the model you can get the translated text using:

        from translate_en_it import translate
        t = translate("<Enter the text you want to translate here!>")

    Check the ``inference.ipynb`` notebook for an example.
