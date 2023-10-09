def find_farthest_distance(N, elves):
    elves.sort()  # Sort the elves' positions in ascending order
    meeting_points = []

    # Calculate the meeting points for each pair of adjacent elves
    for i in range(N - 1):
        meeting_points.append((elves[i] + elves[i + 1]) / 2)

    # Calculate the maximum distance Santa needs to walk
    max_distance = max(meeting_points[0] - elves[0], elves[-1] - meeting_points[-1])

    for i in range(1, len(meeting_points)):
        max_distance = max(max_distance, meeting_points[i] - meeting_points[i - 1])

    return max_distance

# Input
T = int(input())
for case in range(1, T + 1):
    N = int(input())
    elves = list(map(int, input().split()))

    # Calculate and print the farthest distance Santa would need to walk
    farthest_distance = find_farthest_distance(N, elves)
    print(f"Case #{case}: {farthest_distance:.6f}")
