def recursive_dict_compare(all_cfg, other_cfg):
    """
    Recursively compare two dictionaries and return their differences.
    """

    
    # Initialize the result dictionary
    diff = {}

    # Check for keys in dict1 that are not in dict2
    for key in other_cfg:
        if key not in all_cfg:
            diff[key] = other_cfg[key]
        else:
            # If the values are dictionaries, recursively compare them
            if isinstance(all_cfg[key], dict) and isinstance(other_cfg[key], dict):
                nested_diff = recursive_dict_compare(all_cfg[key], other_cfg[key])
                if nested_diff:
                    diff[key] = nested_diff
            # Otherwise, compare the values directly
            elif all_cfg[key] != other_cfg[key]:
                if not(key == "num_classes" and other_cfg[key] is None and all_cfg[key] is not None):
                    diff[key] = other_cfg[key]
                    

    return diff