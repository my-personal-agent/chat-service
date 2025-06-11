def deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merges two dictionaries.

    Values from the override dictionary will overwrite those in the base dictionary.
    If both values are dictionaries, the merge is performed recursively.

    Args:
        base (dict): The base dictionary to merge into.
        override (dict): The dictionary with override values.

    Returns:
        dict: The merged dictionary.
    """
    for key, value in override.items():
        # If both base and override have a dict at this key, merge them recursively
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            base[key] = deep_merge(base[key], value)
        else:
            # Otherwise, override the base value
            base[key] = value
    return base
