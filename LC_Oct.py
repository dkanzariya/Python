class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        n = len(s)
        count = 0
        left = 0
        min_len = float('inf')
        result = ""
        results = []
        final = ""
        for right in range(n):
            if s[right] == '1':
                count += 1

            while count == k:
                if right - left + 1 <= min_len:
                    if right - left + 1 < min_len:
                        results = []
                    min_len = right - left + 1
                    result = s[left:right + 1]
                    results.append(result)
                    # print(results)

                if s[left] == '1':
                    count -= 1
                left += 1
        if results == []:
            return ""
        else:
            final = min(results)
            return final

solution = Solution()
s1 = "110101000010110101"
k1 = 3
print(solution.shortestBeautifulSubstring(s1, k1))  # Output: "11001"

s2 = "1011"
k2 = 2
print(solution.shortestBeautifulSubstring(s2, k2))  # Output: "11"

s3 = "000"
k3 = 1
print(solution.shortestBeautifulSubstring(s3, k3))  # Output: ""