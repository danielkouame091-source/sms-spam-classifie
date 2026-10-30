# SMS Spam Detection: Machine Learning vs Deep Learning

A mini project that compares two approaches to detect spam SMS messages:

| | Machine learning | Deep learning |
|---|---|---|
| Model | Linear SVM | Neural network (Embedding + BiLSTM) |
| Text to numbers | TF-IDF | TextVectorization + Embedding layer |
| Library | scikit-learn | TensorFlow / Keras |

Both models are trained and tested on exactly the same data, then compared
with accuracy and F1-score on the spam class. A Streamlit web app lets anyone
type an SMS and see the answer of both models.

## Dataset

SMS Spam Collection: about 5,500 English SMS messages labelled `ham` (normal)
or `spam`. After removing duplicates, 80% is used for training and 20% for
testing (`random_state=42`, same split for both models).

## Project structure

```
SMS-Spam-ML-vs-DL/
├── app/
│   └── main.py              Streamlit web app
├── data.py                  Loads the data and makes the train/test split
├── train_ml.py              Trains the SVM (machine learning)
├── train_dl.py              Trains the neural network (deep learning)
├── compare.py               Compares both models and draws a chart
├── spam_data.csv            Dataset
├── requirements.txt         Python libraries
└── README.md
```

Files created by the scripts: `svm_model.pkl`, `tfidf_vectorizer.pkl`,
`dl_model.keras`, `results_ml.json`, `results_dl.json`, `comparison.png`.

## How to run

```bash
pip3 install -r requirements.txt
python3 train_ml.py      # trains the SVM
python3 train_dl.py      # trains the neural network
python3 compare.py       # prints the comparison and saves comparison.png
streamlit run app/main.py
```

## Results (test set, 1,034 messages)

| Model | Accuracy | F1-score (spam) |
|---|---|---|
| SVM (machine learning) | 0.9807 | 0.9180 |
| Neural network (deep learning) | run `train_dl.py` | run `train_dl.py` |

The SVM detects 99% of the messages it calls spam correctly (precision 0.99)
but misses 15% of the real spam (recall 0.85).

## Conclusion

The SVM is extremely fast to train (a fraction of a second) and already very
accurate on this small dataset. The neural network needs more time and more
libraries, and its advantage depends on the data size. On short and noisy
texts like SMS, a simple model with good features is a strong baseline.
Fill in the final comparison after running `compare.py`.

## Author

Kouassi Kouame Daniel
