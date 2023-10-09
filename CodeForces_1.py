# Read the number of test cases
t = int(input())
# t = 1
# Iterate through each test case
for _ in range(t):
    # Read the number of teams
    n = int(input())
    # n = 4
    # Read the efficiencies of n-1 teams
    efficiencies = list(map(int, input().split()))
    # efficiencies = [3, -4, 5]
    # Calculate the efficiency of the missing team
    total_efficiency = sum(efficiencies)
    missing_efficiency = -total_efficiency

    # Output the result
    print(missing_efficiency)
