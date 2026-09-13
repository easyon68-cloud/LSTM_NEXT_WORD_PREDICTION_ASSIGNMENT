# ⚡ Quick Start - LSTM Next Word Prediction

> **Get Started in 5 Minutes with Your Notebook!**

---

## 🎯 30-Second Overview

Your **LSTM_Assignment.ipynb** trains a neural network to predict the next word in text using Pride and Prejudice. The notebook has **11 complete steps** and runs in **5-15 minutes** on Google Colab with GPU.

**What it does:**
- 📖 Loads Pride and Prejudice text (130K+ words)
- 🔤 Converts text to numbers and creates sequences
- 🧠 Trains LSTM model on word prediction
- 📊 Visualizes training progress
- 🎯 Makes predictions on new text
- 💾 Saves model for reuse

---

## 🚀 Fastest Start (Google Colab - 5 Min!)

### Step 1: Open Your Notebook
- Go to your Google Colab
- Open **LSTM_Assignment.ipynb**

### Step 2: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 3: Update File Path
In **Step 2** cell, update the path:
```python
file_path = '/content/drive/MyDrive/YOUR_FOLDER/LSTM DATA.txt'

with open(file_path, 'r') as file:
    text_data = file.read()
```

### Step 4: Run All Cells
- Click `Runtime` → `Run all`
- Or run cells one by one (Shift+Enter)

### Step 5: See Results!
```
✓ Model trained
✓ Accuracy calculated  
✓ Predictions generated
✓ Model saved
```

**Total Time:** 10-15 minutes on GPU

---

## 📊 What Your Notebook Does (11 Steps)

### 📋 Step Overview

```
1. Import Libraries
   └─ Load TensorFlow, Keras, NumPy, etc.
   
2. Load Dataset
   └─ Read LSTM DATA.txt (748,151 characters)
   
3. Data Preprocessing
   ├─ Convert to lowercase
   ├─ Clean whitespace
   ├─ Create vocabulary (7,562 words)
   └─ Convert to sequences
   
4. Generate Training Data
   ├─ Create 132,954 training pairs
   ├─ Input: 3 words
   └─ Output: Next word
   
5. Build LSTM Model
   ├─ Embedding Layer (128 dims)
   ├─ LSTM Layer (100 units)
   ├─ Dropout Layer (20%)
   └─ Dense Layer (7,562 classes)
   
6. Train Model
   ├─ 50 epochs
   ├─ Batch size: 64
   ├─ Adam optimizer
   └─ Early stopping enabled
   
7. Evaluate Model
   ├─ Check accuracy
   ├─ Training Accuracy: ~18.86%
   └─ Validation Accuracy: ~14.00%
   
8. Visualize Training
   ├─ Plot loss curves
   ├─ Plot accuracy curves
   └─ Save training_history.png
   
9. Make Predictions
   ├─ Input: Text phrase
   ├─ Output: Next word
   └─ Show confidence
   
10. Generate Sentences
    ├─ Seed text
    ├─ Generate 5 words
    └─ Complete sentence
    
11. Save Model
    ├─ Save lstm_next_word_model.h5
    ├─ Save tokenizer.json
    └─ Save model_config.json
```

---

## 📈 Expected Results

### Performance Metrics
```
Model Size:               7,562 words vocabulary
Training Samples:         132,954 sequences
Embedding Dimensions:     128
LSTM Units:               100
Dropout Rate:             20%

Accuracy:
├── Training: 18.86%
├── Validation: 14.00%
└── (Can improve with more training)

Runtime:
├── Google Colab GPU: 5-15 minutes
├── Google Colab CPU: 30-45 minutes
└── Local GPU: 5-10 minutes
```

### Generated Text Example

```
Input:  "it is a"
Output: "it is a truth universally acknowledged"
```

---

## 💻 Running Locally (If Needed)

### Option 1: Jupyter Notebook

```bash
# Install dependencies
pip install -r requirements.txt

# Start Jupyter
jupyter notebook

# Open LSTM_Assignment.ipynb
# Run cells one by one
```

### Option 2: Command Line

```bash
# Convert notebook to Python script
jupyter nbconvert --to script LSTM_Assignment.ipynb

# Run script
python LSTM_Assignment.py
```

---

## 🔧 Configuration (Pre-Set in Notebook)

Your notebook already has optimal settings:

```python
# Data Configuration
seq_length = 3                    # 3 words predict next
num_words = 5000                  # Top 5000 words

# Model Configuration
embedding_dim = 128               # Word vector size
lstm_units = 100                  # LSTM memory cells
dropout_rate = 0.2                # Overfitting prevention
learning_rate = 0.001             # Adam optimizer rate

# Training Configuration
epochs = 50                        # Training iterations
batch_size = 64                    # Samples per update
validation_split = 0.2            # 20% validation data
```

---

## 🎯 Common Tasks

### Task 1: Run the Notebook
```
1. Open LSTM_Assignment.ipynb
2. Mount Google Drive (Step after imports)
3. Update file path to LSTM DATA.txt
4. Run all cells
5. See results!
```

### Task 2: Use Your Own Text File
```python
# In Step 2, replace path with your file
file_path = '/path/to/your/text.txt'

with open(file_path, 'r') as file:
    text_data = file.read()
```

### Task 3: Change Sequence Length (More Context)
```python
# In Step 4, modify:
seq_length = 5  # was 3
# More context = better predictions
```

### Task 4: Make Predictions on New Text
```python
# After training, use Step 9
result = predict_next_word(model, tokenizer, "it is a")
print(result)  # Shows next word
```

### Task 5: Generate Longer Text
```python
# After training, use Step 10
generated = generate_sentence(model, tokenizer, "it is", num_words=10)
print(generated)
```

---

## 📂 Files Generated After Running

After completing all cells, you'll have:

```
outputs/
├── lstm_next_word_model.h5      # Trained neural network
├── tokenizer.json                # Word vocabulary mapping
├── model_config.json             # Model settings
└── training_history.png          # Performance graphs
```

You can reuse these files to make predictions without retraining!

---

## ✅ Checklist to Get Started

- [ ] Have LSTM_Assignment.ipynb file
- [ ] Have LSTM DATA.txt file (Pride & Prejudice)
- [ ] Open notebook in Google Colab (easiest!)
- [ ] Mount Google Drive
- [ ] Update file path in Step 2
- [ ] Run all cells
- [ ] See model train
- [ ] Check results

---

## 🐛 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| **"File not found"** | Update path in Step 2 |
| **"Out of memory"** | Reduce batch_size to 32 |
| **"Slow training"** | Enable GPU in Colab |
| **"Low accuracy"** | Increase epochs to 100 |
| **"Need more time"** | Reduce dataset size |

---

## 📊 What Happens in Each Step

### Steps 1-2: Setup
- Import libraries ✓
- Load text file ✓

### Steps 3-4: Prepare Data
- Clean text (lowercase, trim spaces)
- Create vocabulary (7,562 unique words)
- Generate 132,954 training pairs

### Steps 5-6: Build & Train
- Create LSTM model (4 layers)
- Compile with Adam optimizer
- Train for 50 epochs with validation

### Steps 7-8: Evaluate & Visualize
- Check accuracy (18.86% train, 14% validation)
- Plot loss and accuracy curves
- Save graphs as PNG

### Steps 9-10: Predict & Generate
- Predict next word from text
- Generate multi-word sequences
- Show results

### Step 11: Save
- Save trained model
- Save vocabulary
- Save configuration

---

## 🎯 Next Actions

### Immediate (After Running)
1. ✅ Run all cells
2. ✅ See model train
3. ✅ Check accuracy
4. ✅ Make predictions

### Short Term (In 1 Hour)
1. Modify hyperparameters
2. Experiment with sequence length
3. Try different seed texts
4. Generate different sentences

### Medium Term (In 1 Day)
1. Train on different text
2. Increase epochs for better accuracy
3. Build web interface
4. Deploy as API

### Long Term (In 1 Week)
1. Add more layers
2. Use pre-trained embeddings
3. Deploy to production
4. Share on GitHub

---

## 📚 Additional Resources

**In This Repo:**
- `README.md` - Complete documentation
- `CONCEPTS_EXPLAINED.md` - Deep dive into LSTM
- `requirements.txt` - Dependencies list

**External:**
- [TensorFlow Docs](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)
- [Project Gutenberg](https://www.gutenberg.org/)

---

## 🚀 Success Indicators

Your notebook is working when you see:

✅ Libraries imported successfully
✅ Data loaded (130K+ words)
✅ Vocabulary created (7,562 words)
✅ Model built with 4 layers
✅ Training starts and loss decreases
✅ Accuracy displayed (18-20%)
✅ Predictions generated
✅ Text sentences generated
✅ Files saved (model, tokenizer, config)

---

## 💡 Pro Tips

1. **First Run**: Execute all cells without modifications
2. **GPU Important**: Enable GPU in Colab for 10x speed
3. **Experiment**: Modify one thing at a time
4. **Save Progress**: Save notebook frequently
5. **Try Different Seeds**: Test with different text phrases

---

## 🎉 Done!

Once you run all cells successfully, you have:
- ✓ Trained LSTM model
- ✓ Text prediction capability
- ✓ Sentence generation
- ✓ Saved model for reuse

**Next:** Read `CONCEPTS_EXPLAINED.md` to understand how it works!

---

**Ready to start? Go run your notebook! 🚀**

*Questions? Check README.md or CONCEPTS_EXPLAINED.md*

