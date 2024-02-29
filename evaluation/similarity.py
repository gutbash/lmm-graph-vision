def calculate_similarity_list(list1, list2) -> float:
    # Ensure the lengths are the same, otherwise, the calculation may not make sense
    if len(list1) != len(list2):
        return 0  # or some other value or handling as desired
    
    matches = sum(1 for x, y in zip(list1, list2) if x == y)
    similarity_percentage = (matches / len(list1)) * 100 if list1 else 100
    return similarity_percentage

def calculate_similarity_dict(dict1, dict2) -> float:
    # Combine keys from both dictionaries to ensure all are considered
    all_keys = set(dict1.keys()).union(dict2.keys())
    total_similarity = 0
    
    for key in all_keys:
        list1 = dict1.get(key, [])
        list2 = dict2.get(key, [])
        union = set(list1).union(list2)
        intersection = set(list1).intersection(list2)
        
        # Calculate list similarity for the key
        list_similarity = (len(intersection) / len(union)) if union else 1  # Handle empty lists case
        total_similarity += list_similarity
    
    # Average the similarities
    overall_similarity = (total_similarity / len(all_keys)) * 100 if all_keys else 100
    return overall_similarity