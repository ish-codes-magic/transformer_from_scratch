# Transformer Model Implementation From Scratch

A complete implementation of the Transformer model from the paper "Attention Is All You Need" (2017) using PyTorch. This project provides a modular, educational implementation that demonstrates how to build transformer networks from individual components.

## 🚀 Features

- **Complete Transformer Architecture**: Full encoder-decoder implementation with multi-head attention, positional encoding, and feed-forward networks
- **Modular Design**: Each component (attention, encoder, decoder, embeddings) is implemented in separate files for better understanding and reusability
- **Educational Focus**: Well-commented code that explains each component's purpose and functionality
- **Training Pipeline**: Complete training loop with proper masking, loss calculation, and optimization
- **Language Translation**: Configured for English to French translation using the OPUS Books dataset
- **Tokenization**: Integrated with HuggingFace tokenizers for text preprocessing

## 📋 Architecture Overview

The implementation consists of the following key components:

### Core Components
- **Multi-Head Attention** (`mha.py`): Scaled dot-product attention mechanism with multiple heads
- **Positional Encoding** (`pos_enc.py`): Sinusoidal positional embeddings to capture sequence order
- **Feed-Forward Network** (`feed_forward.py`): Position-wise fully connected layers with ReLU activation
- **Layer Normalization** (`layer_norm.py`): Custom layer normalization implementation
- **Residual Connections** (`residual_connection.py`): Skip connections around sub-layers

### Model Architecture
- **Input Embeddings** (`embeddings.py`): Token embedding layer with scaling
- **Encoder** (`encoder.py`): Stack of encoder layers with self-attention and feed-forward
- **Decoder** (`decoder.py`): Stack of decoder layers with masked self-attention and cross-attention
- **Projection Layer** (`projection.py`): Final linear layer with log-softmax for output predictions
- **Complete Model** (`model.py`): Full transformer model combining all components

### Training & Data
- **Dataset Handler** (`dataset.py`): Bilingual dataset processing with proper masking
- **Training Script** (`train.py`): Complete training loop with validation and model saving
- **Configuration** (`config.py`): Centralized configuration management

## 🛠️ Technical Specifications

### Model Parameters
- **Model Dimension (d_model)**: 512
- **Number of Heads**: 8
- **Number of Layers**: 6 (encoder) + 6 (decoder)
- **Feed-Forward Dimension**: 2048
- **Dropout Rate**: 0.1
- **Vocabulary Size**: Dynamic (based on dataset)
- **Sequence Length**: 350 tokens

### Training Configuration
- **Batch Size**: 1 (configurable)
- **Learning Rate**: 0.0001
- **Optimizer**: Adam
- **Loss Function**: Cross-Entropy with label smoothing (0.1)
- **Epochs**: 20
- **Dataset**: OPUS Books (English-French)

## 🚀 Usage

### Basic Training
```bash
python train.py
```

### Custom Configuration
Modify `config.py` to adjust model parameters:
```python
def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 30,
        "lr": 0.0001,
        "seq_len": 512,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "fr",
        # ... other parameters
    }
```

### Model Building
```python
from model import build_transformer
from config import get_config

config = get_config()
model = build_transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    src_seq_len=config['seq_len'],
    tgt_seq_len=config['seq_len'],
    d_model=config['d_model']
)
```

## 📊 Model Architecture Details

### Encoder Stack
Each encoder layer contains:
1. **Multi-Head Self-Attention**: Allows the model to attend to different positions
2. **Residual Connection + Layer Norm**: Helps with gradient flow and training stability
3. **Feed-Forward Network**: Adds non-linearity and increases model capacity
4. **Residual Connection + Layer Norm**: Second residual connection

### Decoder Stack
Each decoder layer contains:
1. **Masked Multi-Head Self-Attention**: Prevents attending to future positions
2. **Residual Connection + Layer Norm**
3. **Multi-Head Cross-Attention**: Attends to encoder output
4. **Residual Connection + Layer Norm**
5. **Feed-Forward Network**: Position-wise transformation
6. **Residual Connection + Layer Norm**

### Attention Mechanism
The scaled dot-product attention is computed as:
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

### Training Metrics
- **Loss Function**: Cross-entropy with label smoothing
- **Perplexity**: Tracked during training for model evaluation
- **BLEU Score**: Translation quality measurement
- **Training Time**: Approximately 2-3 hours on modern GPU

## 🔍 Code Structure

```
transformer_from_scratch/
├── config.py              # Configuration management
├── dataset.py             # Data processing and loading
├── decoder.py             # Decoder implementation
├── embeddings.py          # Input embedding layer
├── encoder.py             # Encoder implementation
├── feed_forward.py        # Feed-forward network
├── layer_norm.py          # Layer normalization
├── mha.py                 # Multi-head attention
├── model.py               # Complete transformer model
├── pos_enc.py             # Positional encoding
├── projection.py          # Output projection layer
├── residual_connection.py # Residual connections
├── train.py               # Training script
└── README.md              # This file
```

## 📚 Resources

This implementation is designed to be educational and follows the original paper closely. For deeper understanding, refer to:

1. **Original Paper**: "Attention Is All You Need" (Vaswani et al., 2017)
2. **Harvard's Annotated Transformer**: Detailed walkthrough of the architecture
3. **PyTorch Documentation**: Official documentation for PyTorch modules
4. **Tensor2Tensor**: Google's original implementation

## 🙏 Acknowledgments

- Vaswani et al. for the original Transformer paper
- PyTorch team for the excellent deep learning framework
- HuggingFace for tokenizers and datasets
- Harvard NLP group for the Annotated Transformer tutorial
  
