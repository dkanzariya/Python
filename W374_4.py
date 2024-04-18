MOD = 10**9 + 7
from itertools import permutations

def numberOfSequences(n, sick):
    healthy = []
    for i in range(n):
        if i not in sick:
            healthy.append(i)
            
    if len(healthy) <= 1:
        return 1
        
    seq_count = generateSequences(0, healthy)
    return seq_count % (10**9 + 7)

def generateSequences(i, positions):
    if i == len(positions) - 1:
        return 1
        
    count = 0
    for j in range(i, len(positions)):
        positions[i], positions[j] = positions[j], positions[i]
        count += generateSequences(i+1, positions)
        positions[i], positions[j] = positions[j], positions[i]
    
    return count

    """
    def ways_to_infect(gap_size):
        if gap_size == 0:
            return 1
        return pow(2, gap_size - 1, MOD)
    
    sick = [-1] + sick + [n]
    total_ways = 1

    for i in range(1, len(sick)):
        gap_size = sick[i] - sick[i-1] - 1
        total_ways *= ways_to_infect(gap_size)
        total_ways %= MOD
    
    return total_ways
    """
# Example usage
print(numberOfSequences(5, [0, 4]))  # Output: 4
print(numberOfSequences(4, [1]))     # Output: 3
