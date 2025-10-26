# DL-Journey

Deep Learning Journey — a collection of Jupyter notebooks demonstrating deep learning concepts, from fundamental backpropagation-from-scratch implementations to practical Keras/TensorFlow projects (image classification, transfer learning, NLP, time-series/LSTM, word embeddings, and model tuning).

Repository description: Deep Learning Journey or Practise Deep Learning

## Repository contents (notebooks & data)
This repository is composed mainly of Jupyter Notebooks. The filenames indicate the main topic of each notebook:

- .gitignore
- Admission_Predict_Ver1.1.csv — admissions dataset used by admission prediction notebooks
- LSTM_Next_Word_Prediction.ipynb — LSTM sequence model for next-word prediction / simple language model
- Mnist_multiclass_ANN.ipynb — a multilayer perceptron (ANN) for MNIST digit classification
- backpropagation_classification_from_scratch.ipynb — manual implementation of backpropagation for a classification task (educational)
- backpropagation_regression_from_scratch.ipynb — manual implementation of backpropagation for a regression task (educational)
- customer_churn_prediction.ipynb — model(s) to predict customer churn (classification), likely tabular data + preprocessing
- deep_rnn_vs_lstm_vs_gru.ipynb — comparison of deep RNN, LSTM and GRU architectures and behaviors
- dog-vs-cat.ipynb — binary image classification (dog vs cat) using convolutional / Keras workflows and likely transfer learning
- functional_api_demo.ipynb — demonstration of Keras Functional API (custom architectures, non-sequential models)
- functional_api_multiple_input.ipynb — Keras Functional API example with multiple inputs
- functional_api_multiple_input (1).ipynb — duplicate or variant of the multiple-input notebook (large file present)
- gre-admission-prediction.ipynb — GRE admission prediction (uses Admission_Predict_Ver1.1.csv)
- keras_hyperparameter_tuning.ipynb — Keras model hyperparameter tuning examples (GridSearch/Random/keras-tuner style)
- music_genre_classification_gtzan.ipynb — music genre classification using the GTZAN dataset (audio feature extraction like MFCCs)
- text_classification_imdb50k.ipynb — text classification on IMDB or similar dataset (sentiment classification)
- tmdb_dataset_nlp_perform.ipynb — NLP/performance experiments using TMDB dataset (likely movie overview / metadata)
- transfer_learning_feature_extraction(data_augmentation)_ipynb.ipynb — transfer learning + feature extraction with data augmentation
- transfer_learning_feature_extraction(without_data_augmentation).ipynb — transfer learning + feature extraction without augmentation
- transfer_learning_finetuning.ipynb — transfer learning with fine-tuning the pre-trained backbone
- word2vec.ipynb — training/using Word2Vec embeddings (gensim / Keras examples)

Notes:
- Some notebooks are large (several hundred MB) — these may include saved models, figures, or embedded data outputs.
- There appears to be a duplicate/variation: `functional_api_multiple_input (1).ipynb`. Consider removing/merging duplicates to reduce repo size.

## Quick start / setup

1. Clone the repository:
   git clone https://github.com/Shezan57/DL-Journey.git
   cd DL-Journey

2. Create a Python virtual environment (recommended):
   python3 -m venv venv
   source venv/bin/activate  # macOS / Linux
   venv\Scripts\activate     # Windows

3. Install commonly required packages:
   pip install --upgrade pip
   pip install jupyterlab notebook numpy pandas scikit-learn matplotlib seaborn tensorflow keras pillow opencv-python nltk gensim librosa

Notes on versions and extras:
- TensorFlow 2.x is recommended (e.g., tensorflow>=2.4). Some notebooks may assume Keras is available via `tensorflow.keras`.
- For GPU acceleration, install the appropriate tensorflow package for your CUDA/cuDNN setup (e.g., `pip install tensorflow` or `pip install tensorflow-gpu` when using supported older versions).
- Audio notebooks (music_genre_classification_gtzan.ipynb) may require `librosa` and `soundfile`.
- Word2Vec examples may use `gensim`.
- If you prefer reproducible installs, create a `requirements.txt` from your environment and share it in the repo.

4. Launch Jupyter:
   jupyter lab
   or
   jupyter notebook

Open the notebook you want to run.

## Datasets and external data
- Admission_Predict_Ver1.1.csv is included in the repo and used by GRE/admission notebooks.
- Many notebooks download or load datasets via Keras (MNIST, IMDB) or expect external datasets:
  - GTZAN (music genre) is typically not included — download separately and update paths in the notebook.
  - Dog-vs-cat dataset (or a similar Kaggle dataset) may be required for the dog-vs-cat notebook.
  - TMDB dataset (movie metadata) may be from Kaggle — check notebook comments for download instructions.
- If a notebook expects a dataset that is not present, read the initial cells — most notebooks include instructions on where to download the required data.

## Recommendations & runtime notes
- Use a GPU runtime (Colab or local GPU) for training-heavy notebooks (transfer learning, large CNNs, LSTM with big datasets, Word2Vec on large corpora).
- For reproducibility, restart kernels and clear outputs before running notebooks end-to-end.
- Some notebooks contain exploratory cells, graphs, and saved model outputs that increase the .ipynb size.

## Suggested workflow
- Start with the 'from_scratch' backpropagation notebooks to understand fundamentals.
- Move to MNIST and ANN examples for basic Keras usage.
- Explore functional API notebooks to learn building non-sequential models.
- Try transfer learning notebooks (feature extraction vs fine-tuning) for practical image tasks.
- Explore NLP notebooks (word2vec, LSTM, tmdb, IMDB) for language modeling and classification.
- Use hyperparameter tuning notebook to learn model selection techniques.

## Contributing
- Contributions are welcome. Suggestions:
  - Add a requirements.txt or environment.yml for reproducibility.
  - Add smaller example datasets or links/instructions to download required datasets.
  - Consolidate duplicate notebooks or large outputs (clear output to reduce size).
  - Add a short README per notebook explaining expected datasets/inputs.

## License
- No license file included. Please add a LICENSE if you intend to define terms for reuse.

## Contact
- Repository owner: Shezan57 (GitHub)
- If you want improvements or spot issues, open an issue or submit a PR.

---

I examined the repository structure and crafted this README to make the collection of notebooks easier to navigate. If you want, I can:
- Create and add a requirements.txt or environment.yml.
- Add a smaller per-notebook README with dataset download links and example commands.
- Clean up duplicate notebooks (e.g., `functional_api_multiple_input (1).ipynb`) and reduce repo size by clearing large outputs. Tell me which you'd like next and I'll prepare changes.
