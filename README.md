# lllms_dont_see_age
This repo contains the data and code for the paper "LLMs Do Not See Age: Assessing Demographic Bias in Automated Systematic Reviews" Presented at AACL 2025


# 🧩 Project Structure

This project is organized into modular Python scripts, each handling a specific part of the data processing, summarization, and evaluation workflow.

1. imports_and_models.py

Handles all imports, model initialization, and tokenizer loading.

Includes:

Imports for libraries (transformers, torch, pandas, etc.)

Initialization of:

FactCC model (BertForSequenceClassification)

BGE embedding model (AutoModel)

Tokenizers for both models

Purpose:
Centralizes dependencies and model loading to prevent redundancy across other scripts.

2. text_processing_utils.py

Provides utility functions for text and review data preprocessing.

Functions:

overlength_penalty — computes penalties for summaries exceeding token limits.

calculate_overlength_penalties — batch penalty calculation for multiple records.

stats_summary — computes summary statistics (mean, min, max, std).

unique_reviews — counts unique systematic reviews.

concat_abstracts_by_review — concatenates abstracts under a single review.

avg_word_count — computes the average word count across abstracts.

Purpose:
Basic data cleaning, tokenization checks, and descriptive statistics.

3. merging_utils.py

Contains helper functions for aligning, merging, and standardizing multiple datasets.

Functions:

merge_original_and_generated — merges original abstracts with generated summaries.

standardize_generated_records — standardizes key names in generated records.

merge_generated_and_entities — combines generated summaries with extracted entities.

expand_json_string_fields — expands JSON-like string fields into dictionaries.

merge_entities — merges reference and generated entities across datasets.

Purpose:
Prepares consistent datasets for evaluation and analysis.

4. entity_analysis.py

Implements entity-level analysis and evaluation functions for assessing summary quality.

Functions:

extract_age_entities — uses an LLM to extract age-related entities from text.

compute_ers_hallucination_dss — calculates Entity Retention Score (ERS), hallucinations, and the DSS metric.

omission_overgeneralisation_percent — computes omission and overgeneralisation rates.

per_item_entity_counts — entity count verification per record.

compute_semantic_similarity_and_ers_v2 — evaluates semantic similarity between generated and reference entities.

get_embedding — computes mean-pooled sentence embeddings.

Purpose:
Performs quantitative and semantic entity-level evaluations of generated summaries.

5. summary_generation.py

Contains logic for automated summary generation via language models.

Functions:

generate_summary — prompts a language model to synthesize structured, systematic-review-style summaries from research abstracts (two prompt versions included).

Purpose:
Automates generation of coherent review summaries based on provided abstracts.

6. evaluation_utils.py

Provides data evaluation and conversion utilities.

Functions:

create_gold_generated_df — builds a DataFrame comparing gold (reference) and generated summaries.

sets_to_dicts — recursively converts sets to dictionaries for JSON serialization.

Purpose:
Supports downstream evaluation and export of structured results.

7. graph_prompt.py

Implements structured information extraction using a prompt-based LLM pipeline.

Includes:

SYS_PROMPT — the system prompt for study metadata extraction.

Triples — a Pydantic model defining output schema.

llm and parser initialization.

graphPrompt — runs a LangChain pipeline to extract structured data from academic text.

Purpose:
Extracts key structured metadata (title, author, year, abstract, etc.) from unstructured text.


# How to Use:

🚀 How to Run
1. Set up your environment

Make sure you have Python >= 3.12 installed.
Then install all dependencies listed in requirements.txt

pip install -r requirements.txt

2. Project Import Structure

Ensure your directory looks like this:

project_root/
│
├── imports_and_models.py
├── text_processing_utils.py
├── merging_utils.py
├── entity_analysis.py
├── summary_generation.py
├── evaluation_utils.py
├── graph_prompt.py
└── __init__.py   

3. Example Usage

Here’s an example end-to-end workflow combining the modules.

from imports_and_models import tokenizer
from text_processing_utils import concat_abstracts_by_review
from summary_generation import generate_summary
from merging_utils import merge_original_and_generated
from entity_analysis import compute_ers_hallucination_dss
from evaluation_utils import create_gold_generated_df
