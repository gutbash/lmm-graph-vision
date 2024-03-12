def calculate_similarity_list(list1, list2) -> float:
    """Calculate similarity percentage between two lists using Normalized Levenshtein Distance."""
    distance = levenshtein_distance(list1, list2)
    max_distance = max(len(list1), len(list2))
    if max_distance == 0: # To handle the case where both lists are empty
        return 100.0
    similarity_percentage = ((max_distance - distance) / max_distance) * 100
    return similarity_percentage

def levenshtein_distance(a, b):
    """Calculate the Levenshtein distance between two lists."""
    if not a: return len(b)
    if not b: return len(a)
    # Initialize matrix of zeros
    matrix = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    # Initialize zeroth row and column with indices
    for i in range(len(a) + 1):
        matrix[i][0] = i
    for j in range(len(b) + 1):
        matrix[0][j] = j
    # Populate matrix
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i-1] == b[j-1]:
                cost = 0
            else:
                cost = 1
            matrix[i][j] = min(matrix[i-1][j] + 1,      # Deletion
                               matrix[i][j-1] + 1,      # Insertion
                               matrix[i-1][j-1] + cost) # Substitution
    return matrix[len(a)][len(b)]

def jaccard_index(list1, list2) -> float:
    """
    Calculate the Jaccard Index between two lists.
    Jaccard Index = |Intersection(A, B)| / |Union(A, B)|
    """
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    # Handle the case where both lists are empty, resulting in division by zero
    if not union:
        return 1.0
    jaccard_similarity = len(intersection) / len(union)
    return jaccard_similarity

def calculate_similarity_dict(dict1, dict2) -> float:
    """
    Calculate the overall similarity between two dictionaries based on the Jaccard Index
    of their list values for each key.
    """
    # Combine keys from both dictionaries to ensure all are considered
    all_keys = set(dict1.keys()).union(dict2.keys())
    total_similarity = 0
    
    for key in all_keys:
        list1 = dict1.get(key, [])
        list2 = dict2.get(key, [])
        # Calculate list similarity for the key using the Jaccard Index
        list_similarity = jaccard_index(list1, list2)
        total_similarity += list_similarity
    
    # Average the similarities across all keys
    overall_similarity = (total_similarity / len(all_keys)) * 100 if all_keys else 100
    return overall_similarity
