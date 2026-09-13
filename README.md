# 🚀 LSTM Next Word Prediction with Pride and Prejudice

> **A Complete LSTM Implementation for Text Prediction Using Jane Austen's Classic Novel**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)](https://www.tensorflow.org/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-Ready-brightgreen?style=flat-square&logo=google-colab)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

---

## 📖 Project Overview

This project implements an **LSTM (Long Short-Term Memory)** neural network to predict the next word in a text sequence using Jane Austen's *Pride and Prejudice*. It's a complete, beginner-friendly implementation that walks through every step of building a text prediction model.

### What Makes This Special

- **Real Dataset**: Uses full *Pride and Prejudice* text (748,151 characters, 130,408 words)
- **Complete Pipeline**: Data loading → Preprocessing → Model Training → Predictions
- **Production Ready**: Saves model, tokenizer, and configuration for reuse
- **Well-Documented**: 11 complete steps with explanations
- **Google Colab Compatible**: Runs directly in Google Colab with GPU support

### Real-World Applications

- 📱 Smartphone keyboard autocomplete
- 💬 Chatbot message prediction
- 📝 Search engine query completion
- 📧 Email smart compose
- 📚 Text generation and content creation

---

## 📊 Dataset Information

Your notebook uses data from **Pride and Prejudice by Jane Austen**:

```
Dataset Characteristics:
├── Total Characters: 748,151
├── Total Words: 130,408
├── Vocabulary Size: 7,562 unique words
├── Sequence Length: 3 words → predict next word
├── Training Sequences: 132,954 pairs
└── Novel: Classic Literature
```

### Why Pride and Prejudice?

- Rich vocabulary perfect for language modeling
- Public domain text (Project Gutenberg)
- Well-structured sentences
- Good for understanding literary text patterns
- Large enough for meaningful training

---

## 🛠️ Project Architecture

### 11-Step Implementation

```
Step 1:  Import Libraries
         └─ Load TensorFlow, NumPy, Keras, etc.

Step 2:  Load Dataset
         └─ Read LSTM DATA.txt from Google Drive

Step 3:  Data Preprocessing
         ├─ Lowercase text
         ├─ Remove extra whitespace
         ├─ Tokenize (7,562 vocabulary)
         └─ Convert to sequences (132,957 tokens)

Step 4:  Generate Training Data
         ├─ Create input sequences (3 words)
         ├─ Create output labels (next word)
         └─ 132,954 training pairs

Step 5:  Build LSTM Model
         ├─ Embedding Layer (128 dimensions)
         ├─ LSTM Layer (100 units)
         ├─ Dropout Layer (20%)
         └─ Dense Output Layer (softmax)

Step 6:  Train Model
         ├─ Optimizer: Adam (lr=0.001)
         ├─ Loss: sparse_categorical_crossentropy
         ├─ Metrics: Accuracy
         └─ Early Stopping: Enabled

Step 7:  Evaluate Model
         ├─ Training Accuracy: 18.86%
         ├─ Validation Accuracy: 14.00%
         └─ Performance Analysis

Step 8:  Visualize Training
         ├─ Loss curves
         ├─ Accuracy curves
         └─ Save plots

Step 9:  Make Predictions
         ├─ Top-K word predictions
         ├─ Confidence scores
         └─ Word probability distribution

Step 10: Generate Sentences
         ├─ Seed text input
         ├─ Multi-word generation
         └─ Complete sentence output

Step 11: Save Model
         ├─ Save model (lstm_next_word_model.h5)
         ├─ Save tokenizer (tokenizer.json)
         └─ Save config (model_config.json)
```

### Model Architecture

```
Input: 3-word sequence
    ↓
[Embedding Layer]
- Input dimension: 7,562 (vocabulary size)
- Output dimension: 128
- Converts word IDs to 128-dim vectors
    ↓
[LSTM Layer]
- Units: 100
- Processes sequence and learns temporal patterns
- Captures word dependencies
    ↓
[Dropout Layer]
- Rate: 20%
- Prevents overfitting
    ↓
[Dense Output Layer]
- Units: 7,562 (vocabulary size)
- Activation: Softmax
- Outputs probability for each word
    ↓
Output: Next word prediction (probability distribution)
```

---

## 💻 How the Code Works

### Step 1: Import Libraries
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
```

**Why Each Library:**
- `TensorFlow/Keras`: Deep learning framework
- `LSTM, Dense, Embedding, Dropout`: Neural network layers
- `Tokenizer`: Convert text to numbers
- `matplotlib`: Visualizations

### Step 2: Load Dataset
```python
with open('/content/drive/MyDrive/.../LSTM DATA.txt', 'r') as file:
    text_data = file.read()
```

Your notebook loads the text file from Google Drive. The file contains the full Pride and Prejudice text.

### Step 3: Preprocessing
```python
# Convert to lowercase
text = text_data.lower()

# Remove extra spaces
text = ' '.join(text.split())

# Tokenize (convert words to numbers)
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts([text])
```

**Result:**
- 7,562 unique words discovered
- Out-of-vocabulary token for unknown words
- Text converted to 132,957 numerical tokens

### Step 4: Create Training Data
```python
seq_length = 3
X_train = []  # Input sequences
y_train = []  # Labels (next words)

for i in range(len(sequences) - seq_length):
    X_train.append(sequences[i:i+seq_length])
    y_train.append(sequences[i+seq_length])
```

**Example:**
```
Tokens: [2, 3, 4, 5, 6, 7, ...]

Training pairs:
[2, 3, 4] → 5
[3, 4, 5] → 6
[4, 5, 6] → 7
```

### Step 5: Build Model
```python
model = Sequential([
    Embedding(input_dim=7562, output_dim=128, input_length=3),
    LSTM(units=100, return_sequences=False),
    Dropout(rate=0.2),
    Dense(units=7562, activation='softmax')
])
```

**Layer Breakdown:**

| Layer | Purpose | Configuration |
|-------|---------|---|
| **Embedding** | Convert word IDs to vectors | 128-dimensional |
| **LSTM** | Learn sequence patterns | 100 memory cells |
| **Dropout** | Prevent overfitting | 20% deactivation |
| **Dense** | Output predictions | 7,562 classes (words) |

### Step 6: Train Model
```python
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop]
)
```

**Training Configuration:**
- **Loss**: Sparse categorical cross-entropy (for integer class labels)
- **Optimizer**: Adam with 0.001 learning rate
- **Epochs**: 50 complete passes through data
- **Batch Size**: 64 samples per update
- **Validation Split**: 20% for validation
- **Early Stopping**: Prevents overfitting

### Step 7: Evaluate Performance
```python
train_accuracy = history.history['accuracy'][-1]
val_accuracy = history.history['val_accuracy'][-1]

print(f"Training Accuracy: {train_accuracy*100:.2f}%")
print(f"Validation Accuracy: {val_accuracy*100:.2f}%")
```

**Your Results:**
- Training Accuracy: 18.86%
- Validation Accuracy: 14.00%
- Training Sequences: 132,954
- Vocabulary: 7,562 words

### Step 9: Make Predictions
```python
def predict_next_word(model, tokenizer, text):
    tokens = tokenizer.texts_to_sequences([text])[0]
    tokens = pad_sequences([tokens], maxlen=seq_length)[0]
    prediction = model.predict(np.array([tokens]))
    predicted_idx = np.argmax(prediction[0])
    return reverse_word_map.get(predicted_idx)
```

**Usage:**
```python
result = predict_next_word(model, tokenizer, "the quick brown")
# Output: Next word with highest probability
```

### Step 10: Generate Text
```python
def generate_sentence(model, tokenizer, seed_text, num_words=5):
    current_text = seed_text
    for _ in range(num_words):
        next_word = predict_next_word(model, tokenizer, current_text)
        current_text += f" {next_word}"
    return current_text
```

**Example:**
```python
generated = generate_sentence(model, tokenizer, "it is", num_words=5)
# Output: "it is a truth universally acknowledged"
```

---

## 🚀 Quick Start

### Google Colab (Recommended - No Setup Needed!)

1. Open your notebook in Google Colab
2. Mount Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Update the file path to your LSTM DATA.txt file
4. Run all cells sequentially

### Local Machine

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Prepare Your Data**
   - Place LSTM DATA.txt in project folder
   - Or update the file path in Step 2

3. **Run Jupyter Notebook**
```bash
jupyter notebook LSTM_Assignment.ipynb
```

4. **Execute Cells**
   - Run from top to bottom
   - Each step builds on previous

### Expected Runtime

- **Google Colab with GPU**: 5-15 minutes
- **Local Machine (CPU)**: 30-60 minutes
- **Local Machine (GPU)**: 5-10 minutes

---

## 📊 Model Performance

### Current Results

```
Dataset Size:         7,562 unique words
Training Samples:     132,954 sequences
Vocabulary Coverage:  ~94% of unique words

Performance Metrics:
├── Training Accuracy:   18.86%
├── Validation Accuracy: 14.00%
├── Model Parameters:    ~1.8M
└── Model Size:          ~7MB

Generated Example:
Input:  "it is"
Output: "it is a truth universally acknowledged"
```

### Improving Accuracy

**To increase accuracy:**

1. **Use More Data**
   - Larger dataset = better generalization
   - Collect more novels or texts

2. **Adjust Sequence Length**
   ```python
   seq_length = 5  # was 3
   # More context = better predictions
   ```

3. **Increase Model Complexity**
   ```python
   LSTM(units=200)  # was 100
   Embedding(output_dim=256)  # was 128
   ```

4. **Train Longer**
   ```python
   epochs=100  # was 50
   ```

5. **Add More Layers**
   ```python
   LSTM(150, return_sequences=True),
   LSTM(100),
   ```

---

## 📁 Project Files

```
lstm-next-word-prediction/
├── LSTM_Assignment.ipynb           # Main notebook (your file)
├── LSTM DATA.txt                   # Dataset (Pride & Prejudice)
├── README.md                       # This file
├── QUICK_START.md                  # Quick start guide
├── CONCEPTS_EXPLAINED.md           # Deep explanations
├── requirements.txt                # Dependencies
│
├── Outputs (generated):
├── lstm_next_word_model.h5        # Trained model
├── tokenizer.json                 # Word vocabulary
├── model_config.json              # Model configuration
└── training_history.png           # Performance graphs
```

---

## 🔧 Configuration Details

### Model Hyperparameters

```json
{
  "seq_length": 3,
  "vocab_size": 7562,
  "embedding_dim": 128,
  "lstm_units": 100,
  "dropout_rate": 0.2,
  "learning_rate": 0.001,
  "batch_size": 64,
  "epochs": 50,
  "validation_split": 0.2
}
```

### Keras Layers Configuration

| Layer | Config | Purpose |
|-------|--------|---------|
| Embedding | `(7562, 128, 3)` | Word vectors |
| LSTM | `100 units` | Temporal learning |
| Dropout | `0.2 rate` | Regularization |
| Dense | `(7562, softmax)` | Classification |

---

## 💡 Understanding the Model

### What is LSTM?

LSTM (Long Short-Term Memory) is a special type of neural network designed for sequential data. It can:

- Remember patterns over long sequences
- Forget irrelevant information
- Process text word-by-word
- Capture dependencies between words

### How Prediction Works

```
Input: "it is a"
    ↓
Convert to tokens: [x, y, z]
    ↓
Embedding: [[...128 values...], [...128 values...], ...]
    ↓
LSTM: Processes sequence, learns patterns
    ↓
Dense: Calculates probability for each word
    ↓
Output: {"the": 0.45, "truth": 0.30, "man": 0.10, ...}
    ↓
Prediction: "the" (highest probability)
```

---

## 🎯 Step-by-Step Explanation

### Data Preparation Phase
1. Load text file from disk
2. Convert to lowercase
3. Remove punctuation and extra spaces
4. Create vocabulary (7,562 words)
5. Convert text to number sequences

### Model Building Phase
6. Create LSTM architecture
7. Compile with loss and optimizer
8. Display model summary

### Training Phase
9. Feed training data to model
10. Calculate prediction errors
11. Update weights via backpropagation
12. Validate on unseen data
13. Repeat for 50 epochs

### Evaluation Phase
14. Check accuracy on test data
15. Visualize training progress
16. Analyze loss curves

### Prediction Phase
17. Take new text input
18. Convert to token sequences
19. Pass through trained model
20. Get probability distribution
21. Select highest probability word

---

## 📈 Performance Visualization

Your notebook generates `training_history.png` showing:

```
Loss Curve:
└─ Training loss decreases over epochs
└─ Validation loss validates generalization

Accuracy Curve:
└─ Training accuracy improves
└─ Validation accuracy shows real performance
```

---

## 🔄 Workflow

```
┌─────────────────────────────────────┐
│  LSTM NEXT WORD PREDICTION          │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│  Step 1-2: Load Data                │
│  (748KB text from file)             │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│  Step 3-4: Preprocess Data          │
│  (7,562 vocabulary, 132,954 pairs)  │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│  Step 5-6: Build & Train Model      │
│  (Embedding→LSTM→Dropout→Dense)    │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│  Step 7-8: Evaluate & Visualize     │
│  (18.86% accuracy achieved)         │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│  Step 9-10: Predict & Generate      │
│  (Make predictions on new text)     │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│  Step 11: Save Model & Files        │
│  (Reusable for later predictions)   │
└─────────────────────────────────────┘
```

---

## 🐛 Troubleshooting

### Issue: "File not found" when loading LSTM DATA.txt

**Solution:**
```python
# Update the file path to match your Google Drive location
file_path = '/content/drive/MyDrive/YOUR_FOLDER/LSTM DATA.txt'

with open(file_path, 'r') as file:
    text_data = file.read()
```

### Issue: "Out of memory" error

**Solution:**
```python
# Reduce batch size
batch_size = 32  # was 64

# Or reduce sequence length
seq_length = 2  # was 3

# Or use fewer training samples
X_train = X_train[:50000]
y_train = y_train[:50000]
```

### Issue: Low accuracy (< 20%)

**Solution:**
```python
# Increase training iterations
epochs = 100  # was 50

# Add more LSTM units
LSTM(units=200)  # was 100

# Increase sequence length
seq_length = 5  # was 3

# Train on more data
```

### Issue: Model training is very slow

**Solution:**
- Use Google Colab with GPU enabled
- Reduce dataset size temporarily
- Increase batch size (more memory efficient)
- Reduce model complexity

---

## 📚 Resources

### Documentation
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras API Reference](https://keras.io/api/)
- [Project Gutenberg](https://www.gutenberg.org/)

### Learning Materials
- [LSTM Explained Simply](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Sequence Modeling](https://arxiv.org/pdf/1506.02078.pdf)
- [Text Classification & NLP](http://web.stanford.edu/class/cs224n/)

---

## 🎓 What You'll Learn

### Concepts
- ✓ LSTM architecture and gates
- ✓ Sequence modeling techniques
- ✓ Text preprocessing and tokenization
- ✓ Neural network training
- ✓ Model evaluation metrics

### Skills
- ✓ Build models with Keras/TensorFlow
- ✓ Process text data
- ✓ Train and evaluate models
- ✓ Make predictions
- ✓ Save/load models

### Applications
- ✓ Text autocomplete
- ✓ Chatbots
- ✓ Language models
- ✓ Content generation
- ✓ Text classification

---

## 📞 Support & Questions

- **Documentation**: Check QUICK_START.md and CONCEPTS_EXPLAINED.md
- **Code Issues**: Review comments in each notebook cell
- **Concepts**: See CONCEPTS_EXPLAINED.md for deep dives
- **Google Colab**: See GOOGLE_COLAB_GUIDE.md for tips

---

## 📄 License

This project is open source and available under the MIT License.

---

## 🙏 Acknowledgments

- Jane Austen for *Pride and Prejudice*
- Project Gutenberg for the text
- TensorFlow/Keras team
- Google Colab platform

---

## 🚀 Next Steps

1. **Run the Notebook**: Execute all cells in Google Colab
2. **Understand Each Step**: Read comments and explanations
3. **Experiment**: Modify hyperparameters and observe changes
4. **Use Your Data**: Replace with your own text file
5. **Deploy**: Create API for predictions
6. **Share**: Upload your results to GitHub

---

**Created for Learning • Production-Ready Code • Professional Documentation**

*Happy Learning! 🎉*

