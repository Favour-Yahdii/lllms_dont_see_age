def merge_original_and_generated(original_data, generated_data):
    """
    Merge the abstracts in the original_data eith those in generated_data, merging on common ids.
    Returns a list of dicts with keys: id, original, generated.
    """

    # Build lookup for generated abstracts
    id_to_generated = {str(item['id']): item['abstract'] for item in generated_data if 'id' in item and 'abstract' in item}

    merged = []

    for item in original_data:
        id_str = str(item['id'])
        if 'id' in item and 'abstract' in item:
            merged.append({
                'id': id_str,
                'original': item['abstract'],
                'generated': id_to_generated.get(id_str, "")
            })

    return merged

def standardize_generated_records(records):
    for d in records:
        if 'ID' in d:
            d['id'] = d.pop('ID')  # Rename by popping
        if 'generated_response' in d:
            d['generated_summary'] = d.pop('generated_response')
    return records

def merge_generated_and_entities(generated_data, entities):
    """
    Merge the abstracts in the generated_data with the reference_entities, merging on common ids.
    Returns a list of dicts with keys: id, generated, reference_entities.
    """

    # Build lookup for generated abstracts
    id_to_generated = {str(item['id']): item['extracted_age_entities'] for item in entities if 'id' in item and 'extracted_age_entities' in item}

    merged = []

    for item in generated_data:
        id_str = str(item['id'])
        if 'id' in item and 'generated_summary' in item:
            merged.append({
                'id': id_str,
                'generated': item['generated_summary'],
                'reference_entities': id_to_generated.get(id_str, "")
            })

    return merged

#calculate DSS and ERS for summaries
def expand_json_string_fields(dict_list, keys_to_expand=None):
    """
    For each dict in dict_list, for each key in keys_to_expand, if the value is a JSON string,
    parse it and merge its keys into the parent dict, but exclude the 'id' key from the expansion.
    If keys_to_expand is None, all string values that look like JSON will be expanded.
    Returns a new list of dicts.
    """
    import json
    expanded_list = []
    for d in dict_list:
        new_dict = d.copy()
        for k, v in d.items():
            if (keys_to_expand is None or k in keys_to_expand) and isinstance(v, str):
                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, dict):
                        parsed.pop('id', None)  # Exclude 'id'
                        new_dict.update(parsed)
                except Exception:
                    pass  # Not a JSON string, skip
        expanded_list.append(new_dict)
    return expanded_list


def merge_entities(reference_entities, gen_entities):
    """
    Merge the reference_entities in reference_entities with the generated_entities and related fields in gen_entities, merging on common ids.
    Returns a list of dicts with keys: id, reference_entities, generated_entities, retained_entities, hallucinated_entities, omissions, overgeneralisations.
    Only reference_entities comes from reference_entities; the rest come from gen_entities.
    """
    # Build lookup for reference entities
    id_to_reference = {str(item['id']): item['extracted_age_entities'] for item in reference_entities if 'id' in item and 'extracted_age_entities' in item}

    merged = []
    for item in gen_entities:
        id_str = str(item.get('id', item.get('id')))
        merged.append({
            'id': id_str,
            'reference_entities': id_to_reference.get(id_str, []),
            'generated_entities': item.get('generated_entities', []),
            'retained_entities': item.get('retained_entities', []),
            'hallucinated_entities': item.get('hallucinated_entities', []),
            'omissions': item.get('omissions', []),
            'overgeneralisations': item.get('overgeneralisations', [])
        })
    return merged
