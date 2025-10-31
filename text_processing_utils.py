def overlength_penalty(text: str, max_tokens: int, tokenizer) -> float:
    """
    Computes the overage penalty as:
        max(0, generated_tokens - max_tokens) / max_tokens
    Uses the provided tokenizer to tokenize the text.
    """
    generated_tokens = len(tokenizer.tokenize(text))
    return max(0, generated_tokens - max_tokens) / max_tokens if max_tokens > 0 else 0.0

def calculate_overlength_penalties(records, tokenizer, max_tokens=750):
    """
    For each record, calculate the overlength penalty for 'generated_summary'.
    Returns a dict: id -> overage_penalty
    """
    penalties = {}
    for d in records:
        text = d.get('generated_summary', '')
        id_ = d.get('id')
        if id_ is not None and text:
            penalties[str(id_)] = overlength_penalty(text, max_tokens, tokenizer)
        else:
            penalties[str(id_)] = 0.0
    return penalties


def stats_summary(name, numbers):
    """
    Returns a dict with average, min, max, and std deviation (rounded to 2 decimals) for a list of numbers.
    """
    import numpy as np
    arr = np.array(numbers)
    return {
        f"{name} avg": f"{arr.mean():.2f}",
        f"{name} min": f"{arr.min():.2f}",
        f"{name} max": f"{arr.max():.2f}",
        f"{name} std": f"{arr.std():.2f}"
    }

def unique_reviews(review_set):
    '''
    finds unique reviews in a set of reviews
    args: review_set - list of dicts of sys reviews
    returns len(unique_reviews)
    '''

    unique_reviews = {d['systematicreviewpmid'] for d in review_set if 'systematicreviewpmid' in d}
    return len(unique_reviews)

# concat the values associated with the abstracts for each dictionary into a string delimited by ABSTRACT and an integer relating to the number of that abstract eg 1 for the first abstract

def concat_abstracts_by_review(dicts, review_key='systematicreviewpmid', abstract_key='abstract'):
    """
    Returns a dictionary that maps each unique review to the string of included abstracts,
    where each abstract is delimited by ABSTRACT X where X is the number of the abstract.
    """
    g = defaultdict(list)

    for i in dicts:
        if review_key in i and abstract_key in i:
            g[i[review_key]].append(i[abstract_key])

    result = []

    for review_id, abstracts in g.items():
        concatenated = "\n".join(
            f"ABSTRACT {i+1}: {abstract}" for i, abstract in enumerate(abstracts)
        )
        result.append({'id': review_id, 'abstract': concatenated})

    return result


def avg_word_count(reviews, abstract_key='abstract'):
    word_counts = [
        len(item.get(abstract_key, "").split())
        for item in reviews if abstract_key in item
    ]
    return sum(word_counts) / len(word_counts) if word_counts else 0
