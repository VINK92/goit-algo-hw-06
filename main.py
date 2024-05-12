import networkx as nx
import matplotlib.pyplot as plt
import heapq

G = nx.Graph()
users = ["Alice", "Bob", "Carol", "David", "Eve"]
for user in users:
    G.add_node(user)

connections = [("Alice", "Bob"), ("Alice", "Carol"), ("Bob", "Carol"), ("Bob", "David"), ("Carol", "Eve"), ("David", "Eve")]
G.add_edges_from(connections)

nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, font_size=12, font_weight='bold')
plt.title("Соціальна мережа")
plt.show()


def dfs_paths(graph, start, goal):
    stack = [(start, [start])]
    while stack:
        (vertex, path) = stack.pop()
        for next in set(graph[vertex]) - set(path):
            if next == goal:
                yield path + [next]
            else:
                stack.append((next, path + [next]))

def bfs_paths(graph, start, goal):
    queue = [(start, [start])]
    while queue:
        (vertex, path) = queue.pop(0)
        for next in set(graph[vertex]) - set(path):
            if next == goal:
                yield path + [next]
            else:
                queue.append((next, path + [next]))

def add_weights(graph, weight=1):
    """Додати ваги до ребер графа."""
    for u, v in graph.edges():
        graph[u][v]['weight'] = weight

def dijkstra(graph, start):
    """Алгоритм Дейкстри для знаходження найкоротших шляхів."""
    distances = {node: float('inf') for node in graph.nodes()}
    distances[start] = 0

    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > distances[current_node]:
            continue

        for neighbor, edge_data in graph[current_node].items():
            weight = edge_data['weight']
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances

print("DFS Paths:")
for path in dfs_paths(G, "Alice", "Eve"):
    print(path)


print("\nBFS Paths:")
for path in bfs_paths(G, "Alice", "Eve"):
    print(path)

add_weights(G)

start_vertex = 'Alice'
shortest_distances = dijkstra(G, start_vertex)
print("Найкоротші відстані від вершини", start_vertex, "до інших вершин:")
for vertex, distance in shortest_distances.items():
    print(f"Від {start_vertex} до {vertex}: {distance}")

num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
node_degrees = dict(G.degree())

print("Основні характеристики графа:")
print("Кількість вершин:", num_nodes)
print("Кількість ребер:", num_edges)
print("Ступінь кожної вершини:", node_degrees)