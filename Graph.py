def count_paths_dag(graph):
    n = len(graph)
    paths = [0] * n
    paths[0] = 1

    for node in range(n):
        for neighbor in graph[node]:
            paths[neighbor] += paths[node]

    return int(paths[-1]/2)

# Read input
n = int(input())
graph = []
for i in range(n):
    neighbors = list(map(int, input().split()))
    graph.append(neighbors)

# Calculate and print the number of paths
num_paths = count_paths_dag(graph)
print(num_paths)
