# Input Dataset

### RealNews

Introduced by Zellers et al. in Defending Against Neural Fake News (https://arxiv.org/abs/1905.12616v3)


**RealNews** is a large corpus of news articles from Common Crawl. Data is scraped from Common Crawl, limited to the 5000 news domains indexed by Google News. The authors used the Newspaper Python library to extract the body and metadata from each article. News from Common Crawl dumps from December 2016 through March 2019 were used as training data; articles published in April 2019 from the April 2019 dump were used for evaluation. After deduplication, RealNews is 120 gigabytes without compression.

| Split      | Examples   |
| ---------- | ---------: |
| train      | 13,804,817 |
| validation | 13,855 |

The Google C4's processed version of dataset was downloaded from https://huggingface.co/datasets/allenai/c4. 

# Model Training

Due to limitation of computing resources, only 20,000 samples were used during the training.

Output model size is ~700MB including parameters and vocabularies for the sake of mobility.


# Model Inference

Streamlit's chatbox is used to be an interface with the trained model for inference.