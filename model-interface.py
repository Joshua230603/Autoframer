import streamlit as st
import torch
from transformers import AutoModelForTokenClassification,AutoTokenizer
import pandas as pd

st.set_page_config(page_title="Text Processor", page_icon="🗨️", layout="wide")

# Function to clean text
def clean_text(text):
    if text is None:
        return None
    clean_text = ""
    for word in text.split():
        if word.startswith("##"):
            clean_text += word[2:]
        else:
            if clean_text:
                clean_text += " "
            clean_text += word
    return clean_text

def model(sentence):
    save_path = "trained_model"
    model = AutoModelForTokenClassification.from_pretrained(save_path, return_dict=True)
    tokenizer = AutoTokenizer.from_pretrained(save_path)

    MAX_LEN = 128
    from torch import cuda
    device = 'cuda' if cuda.is_available() else 'cpu'
    # print(device)0
    data = pd.read_csv("ner_datasetreference.csv", encoding='unicode_escape')
    data = data.fillna(method='ffill')
    entities_to_remove = ["B-art", "I-art", "B-eve", "I-eve", "B-nat", "I-nat"]
    data = data[~data.Tag.isin(entities_to_remove)]
    data['sentence'] = data[['Sentence #','Word','Tag']].groupby(['Sentence #'])['Word'].transform(lambda x: ' '.join(x))
    # let's also create a new column called "word_labels" which groups the tags by sentence 
    data['word_labels'] = data[['Sentence #','Word','Tag']].groupby(['Sentence #'])['Tag'].transform(lambda x: ','.join(x))
    label2id = {k: v for v, k in enumerate(data.Tag.unique())}
    id2label = {v: k for v, k in enumerate(data.Tag.unique())}

    inputs = tokenizer(sentence, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors="pt")

    # move to gpu
    ids = inputs["input_ids"].to(device)
    mask = inputs["attention_mask"].to(device)
    # forward pass
    outputs = model(ids, mask)
    logits = outputs[0]

    active_logits = logits.view(-1, model.num_labels) 
    flattened_predictions = torch.argmax(active_logits, axis=1) 

    tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
    token_predictions = [id2label[i] for i in flattened_predictions.cpu().numpy()]
    wp_preds = list(zip(tokens, token_predictions)) # list of tuples. Each tuple = (wordpiece, prediction)

    word_level_predictions = []
    words = []
    for pair in wp_preds:
        if (pair[0].startswith(" ##")) or (pair[0] in ['[CLS]', '[SEP]', '[PAD]']):
            continue
        else:
            word_level_predictions.append(pair[1])
            words.append(pair[0])

    str_rep = " ".join([t[0] for t in wp_preds if t[0] not in ['[CLS]', '[SEP]', '[PAD]']]).replace(" ##", "")

    # Assuming you have word_level_predictions and wp_preds already defined

    # Create DataFrame from word_level_predictions and wp_preds
    df = pd.DataFrame({
        "Word": [t[0] for t in wp_preds if t[0] not in ['[CLS]', '[SEP]', '[PAD]']],
        "Prediction": word_level_predictions
    })

    # Initialize an empty DataFrame to store parsed predictions
    opframe = pd.DataFrame(columns=['person', 'organisation', 'geopolitical', 'geographical', 'time'])

    # Define a list of tags for each entity type
    tag_mappings = {
        'B-gpe': 'geopolitical',
        'I-gpe': 'geopolitical',
        'B-per': 'person',
        'I-per': 'person',
        'B-geo': 'geographical',
        'I-geo': 'geographical',
        'B-tim': 'time',
        'I-tim': 'time',
        'B-org': 'organisation',
        'I-org': 'organisation'
    }
    row_index = 0 
    # Iterate over rows in the DataFrame and parse predictions
    for word, prediction in zip(df['Word'], df['Prediction']):
        # Check if the prediction matches any of the tags
        if word == '.':
                row_index += 1
                continue 
        if prediction in tag_mappings:
            entity_type = tag_mappings[prediction]
            # Check if the DataFrame is empty or if the row with index 0 doesn't exist yet
            if opframe.empty or row_index not in opframe.index:
                # If it's empty or the row doesn't exist, add a new row with empty values
                opframe.loc[row_index] = [''] * len(opframe.columns)
            # Check if the entity type already has a value in the opframe
               
            if opframe.loc[row_index, entity_type] == '':
                opframe.loc[row_index, entity_type] = word
            else:
                # If there's already a value, concatenate the new word with a space
                opframe.loc[row_index, entity_type] += ' ' + word

            

    # Specify columns to apply cleaning function
    columns_to_clean = ['person', 'organisation', 'geopolitical', 'geographical', 'time']

    # Apply cleaning function to specified columns in DataFrame
    opframe[columns_to_clean] = opframe[columns_to_clean].applymap(clean_text)

    # Display cleaned DataFrame
    return opframe



def main():
    st.title("Text File Processor")

    col1, col2, col3 = st.columns([5,1,4])

    with col1:
        uploaded_file = st.file_uploader("Choose a text file", type=["txt"])    
    if uploaded_file is not None:
        file_contents = uploaded_file.read().decode('utf-8')
    else:
        file_contents = None

    # selected_tab = st.sidebar.selectbox("Select Tab", ["Preview", "Table"])

    # if selected_tab == "Preview":
    # elif selected_tab == "Table":
        
    
    if file_contents:

        with col1:
            # st.header("Preview")
            st.text_area("Preview", file_contents, height=140)
    
        with col3: 
            st.header("Table")
            df = model(file_contents)
            st.table(df)
            # tags, words = model(file_contents)
            # tag_to_column = {"B-per": "First Name", "I-per": "Last Name", "B-tim": "Time", "B-org": "Organization", "B-gpe": "Geopolitical Entity", "B-geo": "Country", "O": None}
            # column_dict = {col: [] for col in set(tag_to_column.values()) if col is not None}
            # for tag, word in zip(tags, words):
            #     col_name = tag_to_column.get(tag)
            #     if col_name is not None:
            #         column_dict[col_name].append(word)
            # max_length = max(map(len, column_dict.values()))
            # column_dict = {col: vals + [None] * (max_length - len(vals)) for col, vals in column_dict.items() if any(vals)}
            # ordered_columns = [tag_to_column[tag] for tag in tags if tag in tag_to_column and tag_to_column[tag] is not None]
            # df = pd.DataFrame({col: column_dict[col] for col in ordered_columns})
            # st.table(df)

            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download CSV",data=csv_data,file_name="table.csv",key="download_button")


if __name__ == "__main__":
    main()

