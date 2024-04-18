from collections import Counter

class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        n = len(word)
        count = 0
        
        for i in range(n - k + 1):
            char_count = Counter()
            is_valid = True
            
            for j in range(i, i+k):
                char_count[word[j]] += 1
                
                if char_count[word[j]] > k or (j > i and abs(ord(word[j]) - ord(word[j-1])) > 2):
                    is_valid = False
                    break
                    
            if is_valid:
                count += 1
        
        return count

# Example usage
sol = Solution()
print(sol.countCompleteSubstrings("igigee", 2))  # Output: 3
print(sol.countCompleteSubstrings("aaabbbccc", 3))  # Output: 6
