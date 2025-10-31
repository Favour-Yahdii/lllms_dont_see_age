import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
import numpy as np


class CausalLMScorer:
    def __init__(self, device='cuda:0', max_length=1024, checkpoint='meta-llama/Llama-3.2-3B-Instruct', separator="\n\nAnswer:"):
        self.device = device
        self.max_length = max_length
        self.separator = separator

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)

        # Loss function
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

    def score(self, srcs, tgts, batch_size=4):
        score_list = []
        for i in range(0, len(srcs), batch_size):
            batch_srcs = srcs[i:i+batch_size]
            batch_tgts = tgts[i:i+batch_size]

            batch_inputs = []
            batch_labels = []

            for src, tgt in zip(batch_srcs, batch_tgts):
                # Add separator to target
                target_with_sep = self.separator + " " + tgt

                # Tokenize source and target separately
                source_tokens = self.tokenizer(src, add_special_tokens=False).input_ids
                target_tokens = self.tokenizer(target_with_sep, add_special_tokens=False).input_ids

                # Concatenate
                input_ids = source_tokens + target_tokens

                # Build labels: mask source part
                labels = [-100] * len(source_tokens) + target_tokens

                # Convert to tensor
                input_ids = torch.tensor(input_ids, device=self.device)
                labels = torch.tensor(labels, device=self.device)

                batch_inputs.append(input_ids)
                batch_labels.append(labels)

            # Pad batch
            batch_inputs = nn.utils.rnn.pad_sequence(batch_inputs, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            batch_labels = nn.utils.rnn.pad_sequence(batch_labels, batch_first=True, padding_value=-100)
            attention_mask = (batch_inputs != self.tokenizer.pad_token_id).long()

            batch_inputs = batch_inputs.to(self.device)
            batch_labels = batch_labels.to(self.device)
            attention_mask = attention_mask.to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=batch_inputs, attention_mask=attention_mask)
                logits = outputs.logits

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = batch_labels[:, 1:].contiguous()

                loss = self.loss_fct(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1))
                loss = loss.view(batch_labels.shape[0], -1)

                # Masked average loss over target tokens
                valid_positions = (shift_labels != -100).sum(dim=1)
                loss_per_example = (loss.sum(dim=1) / valid_positions).cpu().numpy()

                # Negative log-likelihood as score
                scores = [-float(x) for x in loss_per_example]
                score_list.extend(scores)
                score_arr = np.array(score_list)

        return {
            "avg": f"{score_arr.mean():.2f}",
            "min": f"{score_arr.min():.2f}",
            "max": f"{score_arr.max():.2f}",
            "std": f"{score_arr.std():.2f}"
        }

    def multi_ref_score(self, srcs, tgts: List[List[str]], agg="mean", batch_size=4):
        ref_nums = [len(x) for x in tgts]
        if len(set(ref_nums)) > 1:
            raise Exception("You have different numbers of references per test sample.")

        ref_num = len(tgts[0])
        score_matrix = []

        for i in range(ref_num):
            curr_tgts = [x[i] for x in tgts]
            scores = self.score(srcs, curr_tgts, batch_size)
            score_matrix.append(scores)

        if agg == "mean":
            score_list = np.mean(score_matrix, axis=0)
        elif agg == "max":
            score_list = np.max(score_matrix, axis=0)
        else:
            raise NotImplementedError

        return list(score_list)

    def test(self, batch_size=3):
        src_list = [
            'This is a very good idea. Although simple, but very insightful.',
            'Can I take a look?',
            'Do not trust him, he is a liar.'
        ]

        tgt_list = [
            "That's stupid.",
            "What's the problem?",
            'He is trustworthy.'
        ]

        print(self.score(src_list, tgt_list, batch_size))
