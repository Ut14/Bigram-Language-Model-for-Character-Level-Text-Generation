
# Bigram Language Model for Character-Level Text Generation

This repository contains a PyTorch implementation of a Bigram Language Model designed for character-level text generation. The model is trained on Shakespearean text to generate stylistically similar outputs.


## Features

- Character-Level Language Modeling: Processes and generates text at the character level.

- Bigram Context: Uses a bigram-based approach to predict the next character based on the previous one.

- Multi-Head Attention: Leverages attention mechanisms for better context understanding.

- Custom Training Pipeline: Includes loss evaluation, batch generation, and model evaluation on both training and validation datasets.

- Text Generation: Generates coherent text sequences based on the learned patterns.


## Installation

Clone the repository:

```bash
git clone https://github.com/username/bigram-language-model.git
cd bigram-language-model
```
Install dependencies:

```bash
pip install torch numpy
```
Add your input text file (input.txt) to the repository root directory. This file should contain the text data to train the model (e.g., Shakespearean text).
## Usage

### 1. Train the Model
        
Run the training script:
```javascript
python bigram.py
```
This will train the model on the provided dataset (input.txt) and periodically log the training and validation loss.

### 2. Generate Text

After training, generate new text using the trained model:

Modify the following code snippet in train.py to set the context and text length:

```javascript
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
```


## Output

### Training Logs

During training, the model periodically logs the training and validation loss:
```bash
step 0: train loss 2.8453, val loss 2.9215
step 500: train loss 1.7523, val loss 1.8952
...
```
The loss decreases as the model learns to predict the next character more accurately.

### Generated Text

The model generates Shakespearean-style text based on the learned patterns. Example:
```bash
First Citizen:
Let us to the Capitol! Away, away!

MENENIUS:
Nay, my good friends, let me speak.

All:
Speak, speak.
```
## Key Components

1. **Hyperparameters**: Configurable parameters like batch_size, block_size, n_embd, and learning_rate for training.

2. **Dataset Preparation**:Encodes the text into integers and prepares train-test splits.

3. **Model Architecture**:
- Token and positional embeddings.
- Multi-head self-attention mechanism.
- Feedforward layers with residual connections and layer normalization.

4. **Training Pipeline**: Implements loss computation and backpropagation using the AdamW optimizer.

5. **Text Generation**: Generates text using the trained model, token-by-token, up to a specified length.


## Requirements

- Python 3.7+
- PyTorch 1.9+
- NumPy

