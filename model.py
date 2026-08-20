"""
NumPy Text Classifier from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - clean_text
import numpy as np

def clean_text(text: str) -> str:
    # Lowercase the text and replace every non-alphabetic character with a space
    cleaned = ''.join(
        char if 'a' <= char <= 'z' else ' '
        for char in text.lower()
    )
    
    # Remove leading and trailing spaces
    return cleaned.strip()

# Step 2 - tokenize
def tokenize(text: str) -> list:
    # Split on any whitespace and remove empty tokens
    return text.split()

# Step 3 - tokenize_corpus
def tokenize_corpus(texts: list) -> list:
    # Clean and tokenize each document while preserving order
    return [tokenize(clean_text(text)) for text in texts]

# Step 4 - split_train_val_test_indices
def split_train_val_test_indices(
    n_samples: int,
    val_fraction: float,
    test_fraction: float,
    seed: int = 0
) -> tuple:
    # Seed NumPy's RNG
    np.random.seed(seed)

    # Create and shuffle a fresh array of indices
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # Calculate split sizes using truncation
    n_val = int(n_samples * val_fraction)
    n_test = int(n_samples * test_fraction)
    n_train = n_samples - n_val - n_test

    # Split in the required order: train, validation, test
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx

# Step 5 - count_word_frequencies
def count_word_frequencies(tokenized_docs: list) -> dict:
    # Count total occurrences of each token across all documents
    frequencies = {}

    for doc in tokenized_docs:
        for token in doc:
            frequencies[token] = frequencies.get(token, 0) + 1

    return frequencies

# Step 6 - build_vocabulary
def build_vocabulary(word_counts: dict, max_size: int) -> dict:
    # Sort by frequency descending, then alphabetically for ties
    sorted_words = sorted(
        word_counts.items(),
        key=lambda item: (-item[1], item[0])
    )

    # Keep only the top max_size words
    sorted_words = sorted_words[:max_size]

    # Assign indices according to rank order
    return {word: index for index, (word, _) in enumerate(sorted_words)}

# Step 7 - tokens_to_bow
def tokens_to_bow(tokens: list, vocab: dict) -> np.ndarray:
    # Create a zero-filled float vector with one entry per vocabulary word
    bow = np.zeros(len(vocab), dtype=float)

    # Count only tokens that are present in the vocabulary
    for token in tokens:
        if token in vocab:
            bow[vocab[token]] += 1.0

    return bow

# Step 8 - corpus_to_bow_matrix
def corpus_to_bow_matrix(tokenized_docs: list, vocab: dict) -> np.ndarray:
    # Convert each document into a BoW vector and stack them into a matrix
    return np.array(
        [tokens_to_bow(doc, vocab) for doc in tokenized_docs],
        dtype=float
    ).reshape(len(tokenized_docs), len(vocab))

# Step 9 - compute_document_frequencies
def compute_document_frequencies(bow_matrix: np.ndarray) -> np.ndarray:
    # Count the number of documents in which each term appears at least once
    return np.sum(bow_matrix > 0, axis=0)

# Step 10 - compute_idf
def compute_idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    # Compute smoothed inverse document frequency
    return np.log((n_docs + 1) / (df + 1)) + 1

# Step 11 - transform_tfidf
def transform_tfidf(bow_matrix: np.ndarray, idf: np.ndarray) -> np.ndarray:
    # Multiply each term count by its corresponding IDF weight
    return bow_matrix * idf

# Step 12 - fit_tfidf
def fit_tfidf(bow_train: np.ndarray) -> np.ndarray:
    # Compute document frequencies from the training BoW matrix
    df = compute_document_frequencies(bow_train)

    # Compute the smoothed IDF using the number of training documents
    return compute_idf(df, bow_train.shape[0])

# Step 13 - sigmoid
def sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable logistic sigmoid
    result = np.empty_like(z, dtype=float)

    positive = z >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-z[positive]))

    exp_z = np.exp(z[~positive])
    result[~positive] = exp_z / (1.0 + exp_z)

    return result

# Step 14 - logistic_predict_proba
def logistic_predict_proba(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    # Compute linear scores and convert them to class-1 probabilities
    scores = X @ w + b
    return sigmoid(scores)

# Step 15 - binary_cross_entropy
def binary_cross_entropy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    l2_lambda: float
) -> float:
    # Clip probabilities to keep logarithms finite
    y_prob = np.clip(y_prob, 1e-15, 1.0 - 1e-15)

    # Mean binary cross-entropy
    bce = -np.mean(
        y_true * np.log(y_prob) +
        (1.0 - y_true) * np.log(1.0 - y_prob)
    )

    # L2 regularization penalty
    l2_penalty = l2_lambda * np.sum(w ** 2) / 2.0

    return float(bce + l2_penalty)

# Step 16 - logistic_gradients
def logistic_gradients(
    X: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    l2_lambda: float
) -> tuple:
    """Compute gradients of BCE+L2 w.r.t. weights and bias for one full batch."""
    
    # Error between predicted probabilities and true labels
    error = y_prob - y_true

    # Gradient of BCE with respect to weights
    dw = (X.T @ error) / X.shape[0]

    # Add L2 regularization gradient
    dw += l2_lambda * w

    # Gradient of BCE with respect to bias
    db = float(np.mean(error))

    return dw, db

# Step 17 - initialize_logistic_params
def initialize_logistic_params(n_features: int) -> tuple:
    # Return zero-initialized weights and a zero bias
    w = np.zeros(n_features, dtype=float)
    b = 0.0

    return w, b

# Step 18 - gradient_descent_step
def gradient_descent_step(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    lr: float,
    l2_lambda: float
) -> tuple:
    # Compute predictions using the current parameters
    y_prob = logistic_predict_proba(X, w, b)

    # Compute loss before the parameter update
    loss = binary_cross_entropy(y, y_prob, w, l2_lambda)

    # Compute gradients
    dw, db = logistic_gradients(X, y, y_prob, w, l2_lambda)

    # Perform one full-batch gradient descent update
    w_new = w - lr * dw
    b_new = b - lr * db

    return w_new, b_new, loss

# Step 19 - train_logistic_regression
def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    l2_lambda: float,
    n_epochs: int
) -> tuple:
    # Initialize weights and bias
    w, b = initialize_logistic_params(X.shape[1])

    # Store the loss from each epoch
    losses = []

    # Run full-batch gradient descent
    for _ in range(n_epochs):
        w, b, loss = gradient_descent_step(
            X, y, w, b, lr, l2_lambda
        )
        losses.append(loss)

    return w, b, losses

# Step 20 - predict_labels
def predict_labels(proba: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert predicted probabilities into hard binary labels.

    Args:
        proba: 1-D array of probabilities in [0, 1], shape (N,).
        threshold: Decision threshold; proba >= threshold maps to 1.

    Returns:
        Integer array of shape (N,) with values in {0, 1}.
    """
    return (proba >= threshold).astype(int)

# Step 21 - confusion_counts
def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    # Compute the four confusion-matrix counts
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return tp, fp, tn, fn

# Step 22 - metrics_from_counts
def metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> dict:
    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1 score
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Accuracy
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy)
    }

# Step 23 - tune_decision_threshold (not yet solved)
# TODO: implement

# Step 24 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 25 - vectorize_texts (not yet solved)
# TODO: implement

# Step 26 - predict_text (not yet solved)
# TODO: implement

# Step 27 - collect_prediction_errors (not yet solved)
# TODO: implement

