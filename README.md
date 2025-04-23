# Autoframer : A BERT-Based Named Entity Recognition App

This project provides a web interface for running Named Entity Recognition (NER) using a fine-tuned BERT model. Upload a text file, and the app will extract entities like names, organizations, locations, and times.

## 📁 Project Structure
. ├── ner_datasetreference.csv # Dataset used for training ├── train_bert.py # Script to fine-tune BERT ├── app.py # Streamlit interface for running the model └── trained_model/ # Directory containing the fine-tuned BERT model


## 🚀 Features

- Upload and analyze `.txt` files
- Entity extraction: person, organization, geopolitical, geographical, and time
- Cleaned, downloadable output in CSV format
- Simple web interface built with Streamlit

## 🛠️ Setup Instructions

```bash
# Clone the repo
git clone https://github.com/Joshua230603/Autoframer.git
cd Autoframer

# (Optional) Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

## 🧪 Train the Model

Make sure ner_datasetreference.csv is present. Then run:

```bash
python model_train.py
```

This will save your fine-tuned model to the trained_model/ directory.

## 💻 Run the App

```bash
streamlit run .py
```
Open your browser and go to http://localhost:8501 to interact with the app.

## 📦 Output

The output table can be downloaded as a CSV with recognized entities organized into columns.





