from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import json
import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
from bartllm import CausalLMScorer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from bert_score import BERTScorer
from rouge import Rouge
import pandas as pd
from sacrebleu import corpus_bleu
from transformers import BertForSequenceClassification, BertTokenizer
import torch.nn.functional as F

model_path = 'manueldeprada/FactCC'

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Load model and tokenizer for embeddings
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
embedding_model = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5")
