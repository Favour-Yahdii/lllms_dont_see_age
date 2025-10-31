def create_gold_generated_df(original_json, generated_json, input_json=None):
    """
    Create a DataFrame with columns: id, gold, generated_summary, input_abstract
    - original_json: json with keys including 'id' and 'abstract'
    - generated_json: path to json with keys including 'id' and 'generated_summary'
    - input_json: (optional) json with keys including 'id' and 'abstract' (the input abstracts)
    Returns: pandas DataFrame with columns ['id', 'gold', 'generated_summary', 'input_abstract']
    """

    # Build lookup for generated summaries
    if generated_json and 'generated_summary' in generated_json[0]:
        id_to_generated = {str(item['id']): item['generated_summary'] for item in generated_json if 'id' in item and 'generated_summary' in item}
    else:
        id_to_generated = {str(item['id']): item['generated_response'] for item in generated_json if 'id' in item and 'generated_response' in item}
    # Build lookup for input abstracts if provided
    id_to_input_abstract = {}
    if input_json is not None:
        id_to_input_abstract = {str(item.get('id')): item.get('abstract', '') for item in input_json if 'id' in item}

    # Build dataframe rows
    rows = []
    for item in original_json:
        id_str = str(item.get('id'))
        gold = item.get('abstract', '')
        generated_summary = id_to_generated.get(id_str, '')
        input_abstract = id_to_input_abstract.get(id_str, '') if input_json is not None else ''
        rows.append({'id': id_str, 'gold': gold, 'generated_summary': generated_summary, 'input_abstract':input_abstract})
    return pd.DataFrame(rows)

def sets_to_dicts(obj):
    if isinstance(obj, set):
        return {str(k): True for k in obj}
    elif isinstance(obj, dict):
        return {k: sets_to_dicts(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sets_to_dicts(v) for v in obj]
    else:
        return obj
