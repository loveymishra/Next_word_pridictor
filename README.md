Next Word Predictor using LSTM
This project demonstrates how to build a simple next-word prediction model using Long Short-Term Memory (LSTM) networks. The model is trained on subtitles from the Friends TV show, enabling it to predict the next word in a sentence based on the context of prior words. By exploring this project, you'll see how deep learning can be applied to natural language processing tasks in a fun and engaging way.
Table of Contents

Usage
Data
Model Architecture
Testing the Model
Results
Notes
Contributing

Usage

Environment Setup:

Ensure you have Python installed (version 3.6 or higher recommended).
Install the required libraries using pip:pip install tensorflow keras numpy




Running the Notebook:

Open the Jupyter notebook Next_word_prediction_github_notebook.ipynb in your preferred environment (e.g., Jupyter Notebook or JupyterLab).
Execute the cells in sequence to preprocess the data, train the model, and test predictions.



Data
The model is trained on subtitles from 17 episodes of the Friends TV show. These subtitles were sourced from TVsubtitles.net in SRT format and converted to plain text using Happy Scribe's SRT-to-Text tool.
Preprocessing Steps:

Combining Files: Multiple subtitle files are merged into a single text corpus.
Cleaning: Emojis and lines with fewer than two words are removed to ensure meaningful input.
Filtering: Repeated lines (appearing more than four times) are excluded to reduce redundancy, such as "laughter" or "applause."

Model Architecture
The LSTM-based model is designed to learn sequential patterns in text. Its architecture includes:

Embedding Layer: Transforms word indices into dense vectors (100 dimensions) for semantic representation.
LSTM Layers: Two stacked LSTM layers with 150 units each. The first layer returns sequences to pass context to the second layer, enhancing pattern recognition.
Dense Layer: A fully connected layer with softmax activation, outputting probabilities across the vocabulary.

The model uses the Adam optimizer and categorical cross-entropy loss, optimized for multi-class prediction tasks.
Testing the Model
Once trained, the model can predict the next word for given phrases. Example predictions include:

Input: "good" → Predicted: "night"
Input: "where have you" → Predicted: "been"
Input: "i love you" → Predicted: "too"

To test your own inputs, edit the sample phrases in the notebook's testing section and run the cells.
Results
The model was trained for 50 epochs initially, with additional fine-tuning sessions. It achieved a final accuracy of approximately 67%. While effective for Friends-style dialogue, its performance may vary on other text types due to the dataset's specificity.
Notes

The model’s training data is limited to Friends subtitles, so it may not generalize well to other domains (e.g., formal or technical text).
Prediction quality can be enhanced by expanding the dataset or adjusting hyperparameters like LSTM units or training epochs.

Contributing
Found a bug or have an idea to improve the project? Feel free to open an issue or submit a pull request. Contributions are always welcome!
