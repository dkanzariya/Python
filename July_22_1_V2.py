def count_divisible_pairs(N, A, K):
  """
  Counts the number of pairs (X, Y) in the array A such that (X + Y) is divisible by K.

  Args:
    A: A list of integers.
    K: The divisor.

  Returns:
    The number of pairs (X, Y) in the array A such that (X + Y) is divisible by K,
    modulo 10^9 + 7.
  """

  # Create a hash table to store the remainders of the elements of the array A
  # when divided by K.
  N = len(A)
  remainder_counts = {}
  myset = set()
  for element in A:
    remainder = element % K
    myset.add(element)
    if remainder in remainder_counts and element in myset:
      remainder_counts[remainder] += 1
    else:
      remainder_counts[remainder] = 1

  # Iterate over the array A and count the number of pairs (X, Y) such that
  # (X + Y) is divisible by K.
  count_of_pairs = 0
  for element in A:
    remainder = (K - element % K) % K
    if remainder in remainder_counts:
      count_of_pairs += remainder_counts[remainder]
      # if element > remainder_counts[remainder]:
        #count_of_pairs -= 1

  # Return the count of pairs modulo 10^9 + 7.
  return count_of_pairs % (10**9 + 7)


# Example usage:

# A = [0, 1, 2, 3]
# K = 2

# A = [1, 1, 1]
# K = 1

# count_of_pairs = count_divisible_pairs(A, K)

# print(count_of_pairs)  # 8
'''
def count_pairs_divisible_by_K(N, A, K):
    # Create a dictionary to count remainders
    remainder_count = {}
    count = 0

    for num in A:
        rem = num % K
        comp = (K - rem) % K

        if comp in remainder_count:
            count += remainder_count[comp] * remainder_count[rem]

        if rem == comp:
            count += (remainder_count[rem] * (remainder_count[rem] - 1)) // 2

        remainder_count[rem] = remainder_count.get(rem, 0) + 1

    return count % (10 ** 9 + 7)


# Sample Input
N = 4
A = [0, 1, 2, 3]
K = 2
print(count_pairs_divisible_by_K(N, A, K))  # Output: 8

# N = 3
# A = [1, -1, 0]
# K = 2
# print(count_pairs_divisible_by_K(N, A, K))  # Output: 5
#
# N = 3
# A = [1, 1, 1]
# K = 1
# print(count_pairs_divisible_by_K(N, A, K))  # Output: 1
'''