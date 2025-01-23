import os

import gdown
import streamlit as st
import torch
import torchtext

import aux

SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
max_seq_len = 30
seed = 0
temperature = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


loading = st.title("Loading Model...")

import glob
import time

pattern = "best-val-lstm_lm.*.part"
downloading_parts = glob.glob(pattern)

if not os.path.exists("best-val-lstm_lm.pt") and len(downloading_parts) == 0:
    gdown.download(id="16QOuo_XGBRakQZ9a-Nywwy28z0o6oBlH", output="best-val-lstm_lm.pt", quiet=False)
    time.sleep(5)

if os.path.exists("best-val-lstm_lm.pt"):
    tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
    checkpoint = torch.load("best-val-lstm_lm.pt", weights_only=False, map_location=device)
    model = aux.LSTMLanguageModel(
        checkpoint["vocab_size"], checkpoint["emb_dim"], checkpoint["hid_dim"], checkpoint["num_layers"], checkpoint["dropout_rate"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    vocab = checkpoint["vocab"]
else:
    st.error("Error: Model file not found.")
    st.stop()

loading.empty()


st.title("🤔 Chatbot!?")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Enter a prompt to start the text generation."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    generation = aux.generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed)
    st.session_state.messages.append({"role": "assistant", "content": (" ".join(generation))})
    st.chat_message("assistant").write((" ".join(generation)))
