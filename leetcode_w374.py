from typing import List

class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()
        added_coins = 0
        current_max_reachable = 0

        for coin in coins:
            while coin > current_max_reachable + 1 and current_max_reachable < target:
                # Add a new coin
                current_max_reachable += (current_max_reachable + 1)
                added_coins += 1
            current_max_reachable += coin
            if current_max_reachable >= target:
                break

        while current_max_reachable < target:
            current_max_reachable += (current_max_reachable + 1)
            added_coins += 1

        return added_coins

# Example usage
# sol = Solution()
# print(sol.minimumAddedCoins([1, 4, 10], 19)) # Output: 2
# print(sol.minimumAddedCoins([1, 4, 10, 5, 7, 19], 19)) # Output: 1
# print(sol.minimumAddedCoins([1, 1, 1], 20)) # Output: 3


def sum(a, b):
    return a + b