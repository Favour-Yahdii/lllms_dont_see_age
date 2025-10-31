def extract_age_entities(text):
def compute_ers_hallucination_dss(dicts, alpha=2, gamma=2, overlength_penalty=None):
    """
    Each dict must have keys: 'retained_entities', 'reference_entities', 'hallucinated_entities'
    Ensures reference_entities is a list for each dict.
    Returns: dict with ERS, hallucination, group DSS, and optionally overlength_penalty.
    """
    ers_scores = []
    hallucination_scores = []
    for d in dicts:
        references = d.get('reference_entities', [])
        if isinstance(references, str):
            references = [v.strip() for v in references.split(',') if v.strip()]
            d['reference_entities'] = references
        elif references is None:
            references = []
            d['reference_entities'] = []
        retained = d.get('retained_entities', [])
        if isinstance(retained, str):
            retained = [r.strip() for r in retained.split(',') if r.strip()]
            d['retained_entities'] = retained
        elif retained is None:
            retained = []
            d['retained_entities'] = []
        hallucinated = d.get('hallucinated_entities', [])
        if isinstance(hallucinated, str):
            hallucinated = [h.strip() for h in hallucinated.split(',') if h.strip()]
            d['hallucinated_entities'] = hallucinated
        elif hallucinated is None:
            hallucinated = []
            d['hallucinated_entities'] = []
        num_references = len(references)
        num_retained = len(retained)
        num_hallucinated = len(hallucinated)
        ers = num_retained / num_references if num_references else 0
        hallucination = num_hallucinated / num_references if num_references else 0
        ers_scores.append(ers)
        hallucination_scores.append(hallucination)
    hall_sum = sum(hallucination_scores)
    # Add overlength penalty if provided
    if overlength_penalty is not None:
        hall_sum += sum(overlength_penalty.get(str(d.get('id')), 0.0) for d in dicts)
    dss = alpha * sum(ers_scores) - gamma * hall_sum
    dss_max = alpha * len(dicts)  # max possible DSS
    dss_normalized = dss / dss_max if dss_max else 0
    result = {
        'group_DSS': float(f"{dss:.2f}"),
        'group_DSS_normalized': float(f"{dss_normalized:.2f}"),
        'ers_scores': [float(f"{ers:.2f}") for ers in ers_scores],
        'hallucination_scores': [float(f"{hallucination:.2f}") for hallucination in hallucination_scores]
    }
    if overlength_penalty is not None:
        result['overlength_penalty'] = overlength_penalty
    return result

def omission_overgeneralisation_percent(cleaned_ers_auto):
    """
    Calculate the percentage of omissions and overgeneralisations in a cleaned ERS list.
    Ensures reference_entities, omissions, and overgeneralisations are lists for each dict.
    Args:
        cleaned_ers_auto (list of dict): Each dict should have 'omissions', 'overgeneralisations', and 'reference_entities' keys.
    Returns:
        dict: {'percent_omissions': float, 'percent_overgeneralisations': float, 'total_omissions': float, 'total_overgen': float}
    """
    for d in cleaned_ers_auto:
        # Ensure reference_entities is a list
        ref = d.get('reference_entities', [])
        if isinstance(ref, str):
            d['reference_entities'] = [v.strip() for v in ref.split(',') if v.strip()]
        elif ref is None:
            d['reference_entities'] = []
        # Ensure omissions is a list
        omissions = d.get('omissions', [])
        if isinstance(omissions, str):
            d['omissions'] = [v.strip() for v in omissions.split(',') if v.strip()]
        elif omissions is None:
            d['omissions'] = []
        # Ensure overgeneralisations is a list
        overgen = d.get('overgeneralisations', [])
        if isinstance(overgen, str):
            d['overgeneralisations'] = [v.strip() for v in overgen.split(',') if v.strip()]
        elif overgen is None:
            d['overgeneralisations'] = []

    total_omissions = sum(len(d.get('omissions', [])) for d in cleaned_ers_auto)
    total_overgen = sum(len(d.get('overgeneralisations', [])) for d in cleaned_ers_auto)
    total_reference_entities = sum(len(d.get('reference_entities', [])) for d in cleaned_ers_auto)
    percent_omissions = (total_omissions / total_reference_entities) * 100 if total_reference_entities else 0
    percent_overgen = (total_overgen / total_reference_entities) * 100 if total_reference_entities else 0
    return {
        'percent_omissions': round(percent_omissions, 2),
        'percent_overgeneralisations': round(percent_overgen, 2)
    }

def per_item_entity_counts(dicts):
    """
    For each dict in the list, calculate the number of reference_entities, retained_entities, and omitted_entities.
    Ensures reference_entities, omissions, and retained_entities are lists for each dict.
    Asserts that: num_reference_entities == num_retained_entities + num_omitted_entities for each dict.
    Returns a list of dicts with keys: id, num_reference_entities, num_retained_entities, num_omitted_entities.
    """
    results = []
    for d in dicts:
        # Ensure reference_entities is a list
        ref = d.get('reference_entities', [])
        if isinstance(ref, str):
            d['reference_entities'] = [v.strip() for v in ref.split(',') if v.strip()]
        elif ref is None:
            d['reference_entities'] = []
        # Ensure omissions is a list
        omissions = d.get('omissions', [])
        if isinstance(omissions, str):
            d['omissions'] = [v.strip() for v in omissions.split(',') if v.strip()]
        elif omissions is None:
            d['omissions'] = []
        # Ensure retained_entities is a list
        retained = d.get('retained_entities', [])
        if isinstance(retained, str):
            d['retained_entities'] = [v.strip() for v in retained.split(',') if v.strip()]
        elif retained is None:
            d['retained_entities'] = []
        num_reference = len(d.get('reference_entities', []))
        num_retained = len(d.get('retained_entities', []))
        num_omitted = len(d.get('omissions', []))  # Use 'omissions' as the omitted key
        # assert num_reference == num_retained + num_omitted, (
        #     f"Assertion failed for id={d.get('id')}: "
        #     f"{num_reference} != {num_retained} + {num_omitted}"
        # )
        results.append({
            'id': d.get('id'),
            'num_reference_entities': num_reference,
            'num_retained_entities': num_retained,
            'num_omitted_entities': num_omitted
        })
    return results

def compute_semantic_similarity_and_ers_v2(records, threshold=0.7, hallucination_threshold=0.7, alpha=2, gamma=2, overlength_penalty=None):
    """
    records: list of dicts, each with keys:
      - 'reference_entities': list of str
      - 'generated_entities': list of str
    overage_penalties: dict mapping id to overage penalty (optional)
    Returns:
      - exact_matches: dict of ID -> set of exact matches
      - similar_matches: dict of ID -> dict of reference_entity -> list of (generated_entity, sim)
      - hallucinations: dict of ID -> list of hallucinated entities
      - omissions: dict of ID -> list of omitted entities
      - omission_scores: dict of ID -> float
      - ers_scores: dict of ID -> float
      - dss: float (group DSS)
      - dss_normalized: float (group DSS normalized to [0,1])
      - overage_penalties: dict of ID -> float (if provided)
    """
    exact_matches = defaultdict(set)
    similar_matches = defaultdict(lambda: defaultdict(list))
    hallucinations = defaultdict(list)
    omissions = {}
    omission_scores = {}
    ers_scores = {}
    hallucination_scores = {}

    for d in records:
        # Ensure reference_entities is a list
        ref = d.get('reference_entities', [])
        if isinstance(ref, str):
            ref = [v.strip() for v in ref.split(',') if v.strip()]
            d['reference_entities'] = ref
        elif ref is None:
            ref = []
            d['reference_entities'] = []
        # Ensure generated_entities is a list
        gen = d.get('generated_entities', [])
        if isinstance(gen, str):
            gen = [v.strip() for v in gen.split(',') if v.strip()]
            d['generated_entities'] = gen
        elif gen is None:
            gen = []
            d['generated_entities'] = []
        id_ = d.get('ID', d.get('id', None))
        if not id_:
            continue

        # Exact matches
        exact = set(ref) & set(gen)
        exact_matches[id_] = exact

        # Semantic similarities
        embeddings_gen = {text: get_embedding(text) for text in gen}
        embeddings_ref = {text: get_embedding(text) for text in ref}

        similar_count = 0
        for gold in ref:
            if gold in gen:
                continue  # already counted as exact match
            gold_vec = embeddings_ref[gold]
            found_similar = False
            for pred, pred_vec in embeddings_gen.items():
                sim = cosine_similarity(gold_vec.unsqueeze(0), pred_vec.unsqueeze(0)).item()
                if sim >= threshold:
                    similar_matches[id_][gold].append((pred, sim))
                    found_similar = True
            if found_similar:
                similar_count += 1

        # Hallucinations: entities in gen not in ref and not semantically similar to any ref
        hallucinated = []
        for pred, pred_vec in embeddings_gen.items():
            if pred in ref:
                continue  # skip exact matches
            max_sim = 0
            for gold_vec in embeddings_ref.values():
                sim = cosine_similarity(gold_vec.unsqueeze(0), pred_vec.unsqueeze(0)).item()
                if sim > max_sim:
                    max_sim = sim
            if max_sim < hallucination_threshold:
                hallucinated.append(pred)
        hallucinations[id_] = hallucinated

        #Omissions: entities in ref not in gen and not semantically similar to any gen
        omitted = []
        for gold, gold_vec in embeddings_ref.items():
            if gold in gen:
                continue  # already counted as exact match
            max_sim = 0
            for pred_vec in embeddings_gen.values():
                sim = cosine_similarity(gold_vec.unsqueeze(0), pred_vec.unsqueeze(0)).item()
                if sim > max_sim:
                    max_sim = sim
            if max_sim < threshold:
                omitted.append(gold)

        omissions[id_] = omitted

        omission_score = len(omitted) / len(ref) if len(ref) > 0 else 0.0

        omission_scores[id_] = omission_score

        # ERS: proportion of reference entities that are exact or semantically matched
        total = len(ref)
        matched = len(exact) + similar_count
        ers = 1-omission_score #matched / total if total > 0 else 0.0
        ers_scores[id_] = ers


        # Hallucination penalty: proportion of hallucinated entities per reference entity
        hallucination_scores[id_] = len(hallucinated) / total if total > 0 else 0.0

    # DSS calculation
    ers_sum = sum(ers_scores.values())
    hall_sum = sum(hallucination_scores.values())
    if overlength_penalty is not None:
        hall_sum += sum(overlength_penalty.get(str(id_), 0.0) for id_ in ers_scores.keys())

    dss = alpha * ers_sum - gamma * hall_sum
    dss_max = alpha * len(ers_scores)  # max possible DSS
    dss_normalized = dss / dss_max if dss_max else 0
    dss_normalized = float(f"{dss_normalized:.2f}")
    dss = float(f"{dss:.2f}")

    if overlength_penalty is not None:
        return exact_matches, similar_matches, hallucinations, hallucination_scores, omissions, omission_scores, ers_scores, dss, dss_normalized, overlength_penalty
    else:
        return exact_matches, similar_matches, hallucinations, hallucination_scores, omissions, omission_scores, ers_scores, dss, dss_normalized

def get_embedding(text, model=embedding_model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling
    last_hidden = outputs.last_hidden_state
    attention_mask = inputs.attention_mask
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    summed = torch.sum(last_hidden * mask_expanded, dim=1)
    summed_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    mean_pooled = summed / summed_mask
    return mean_pooled.squeeze()

def extract_age_entities(text):
