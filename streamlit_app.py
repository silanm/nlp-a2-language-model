import glob
import os

import gdown
import streamlit as st
import torch
import torchtext

import aux

SEED = 1234
MAX_SEQ_LEN = 30
TEMPERATURE = 0.5

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Streamlit

# --Experimental-----------------------------------------------------------------------------------------------------------------------
loading = st.title("Loading Model...")

# Check if model file exists or being downloaded
pattern = "best-val-lstm_lm.*.part"
downloading_parts = glob.glob(pattern)

# Download model file if it doesn't exist
if not os.path.exists("best-val-lstm_lm.pt") and len(downloading_parts) == 0:
    gdown.download(id="16QOuo_XGBRakQZ9a-Nywwy28z0o6oBlH", output="best-val-lstm_lm.pt", quiet=False)

# Load model
if os.path.exists("best-val-lstm_lm.pt"):
    tokenizer = torchtext.data.utils.get_tokenizer("basic_english")  # tokenizer used to preprocess the input text
    checkpoint = torch.load("best-val-lstm_lm.pt", weights_only=False, map_location=device)  # load model checkpoint
    model = aux.LSTMLanguageModel(
        checkpoint["vocab_size"], checkpoint["emb_dim"], checkpoint["hid_dim"], checkpoint["num_layers"], checkpoint["dropout_rate"]
    )  # initialize model
    model.load_state_dict(checkpoint["model_state_dict"])  # load model state dictionary
    model.to(device)
    model.eval()  # set model to evaluation mode
    vocab = checkpoint["vocab"]  # load vocabulary
else:
    st.error("Error: Model file not found.")
    st.stop()

loading.empty()
# -----------------------------------------------------------------------------------------------------------------------------------


st.title("🤔 Chatbot!?")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Enter a prompt to start the text generation."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    # Append the user's prompt to the chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Generate response from the model based on the user's prompt
    generation = aux.generate(prompt, MAX_SEQ_LEN, TEMPERATURE, model, tokenizer, vocab, device, SEED)

    # Append the generated response to the chat
    st.session_state.messages.append({"role": "assistant", "content": (" ".join(generation))})
    st.chat_message("assistant").write((" ".join(generation)))
