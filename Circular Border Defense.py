def find_starting_index(N, base_army, lost_army):
    total_soldiers = sum(base_army)
    current_soldiers = 0

    for i in range(N):
        # Check if we have enough soldiers to travel to the next base
        current_soldiers = 0
        current_soldiers = base_army[i] - lost_army[i] + base_army[i+1]

        if current_soldiers < lost_army[i]:
            continue  # Not enough soldiers to continue the journey

        # Calculate the soldiers left after reaching the next base

        # Check if we have completed a full circle and can return to the starting base
        if current_soldiers >= base_army[i]:
            return i

    return -1  # Cannot complete a full circle

# Input the number of army bases
N = 5

# Input the army base strengths and lost soldiers
base_army = [1, 2, 3, 4, 5]
lost_army = [3, 4, 5, 1, 2]

# Find the starting index
starting_index = find_starting_index(N, base_army, lost_army)

# Print the result
print(starting_index)
