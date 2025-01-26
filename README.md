# Word-Embeddings-Generative-AI-NLP
Training Word Embeddings over a corpus of text from Lord of the Rings


## **Project Overview**
This project explores the development and utilization of **word embeddings**, a key technique in Natural Language Processing (NLP). The goal is to train embeddings using **word2vec** and **neural networks**, analyze relationships between words, and experiment with pretrained embeddings such as **GloVe**.

---

## **Summary**
### **Tasks Covered**
1. **Implementing Word2Vec (SkipGram Model)**
   - Train word embeddings using a **custom SkipGram implementation**.
   - Use **The Fellowship of the Ring** as the dataset.
   - Implement **Negative Sampling** to improve model efficiency.

2. **Learning Embeddings with Neural Networks**
   - Frame word prediction as a **classification problem**.
   - Train a simple **feedforward neural network** to learn word embeddings.
   - Implement **softmax activation** and backpropagation.

3. **Playing with Word Vectors**
   - Use **Cosine Similarity** to measure word relationships.
   - Train embeddings on **context windows** and compare performance.
   - Apply **pretrained GloVe embeddings** and analyze similarities.

4. **Visualizing Embeddings**
   - Reduce high-dimensional embeddings using **Singular Value Decomposition (SVD)**.
   - Plot word embeddings to observe their semantic relationships.
   - Experiment with **word analogies** using vector arithmetic.

5. **Bias in Word Embeddings**
   - Explore how biases are encoded in word vectors.
   - Analyze gender bias using word similarity metrics.
   - Discuss implications of **implicit biases in AI models**.

---

## **Technologies Used**
- **Python**: Core programming language.
- **NumPy**: Mathematical operations and vectorized computations.
- **Matplotlib & Seaborn**: Data visualization.
- **NLTK**: Tokenization and text preprocessing.
- **Scikit-Learn**: Dimensionality reduction using **SVD**.
- **Gensim**: Pretrained word embeddings (GloVe).

---

## **Architecture**
1. **Dataset Preparation**
   - Preprocess text from *The Fellowship of the Ring*.
   - Tokenize and clean text data.
   - Generate **(context, target) pairs** for training.

2. **SkipGram Model Implementation**
   - Initialize embedding matrices (`W` and `C`).
   - Implement **Negative Sampling**.
   - Train using **gradient descent** and **stochastic updates**.

3. **Neural Network-based Word2Vec**
   - Represent words using **one-hot encoding**.
   - Train a **feedforward neural network** to predict target words.
   - Extract learned embeddings from trained weights.

4. **Evaluation & Visualization**
   - Compute **Cosine Similarity** between word embeddings.
   - Visualize relationships using **2D projections**.
   - Compare embeddings trained from scratch vs. **pretrained GloVe**.

5. **Bias & Ethical Considerations**
   - Explore how word embeddings reflect **gender bias**.
   - Discuss ethical concerns in AI and NLP.

---


## **Key Features**
- **Train custom word embeddings** using SkipGram & Negative Sampling.
- **Develop a simple neural network** for learning embeddings.
- **Visualize word vectors** using dimensionality reduction techniques.
- **Compare model-based embeddings with pretrained GloVe vectors**.
- **Analyze and discuss biases in word embeddings**.



