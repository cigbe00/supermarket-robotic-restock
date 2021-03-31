import pygame
import heapq
from Node import Node
from Graph import Graph
import random


def compute_next_node(graph, node):
    next_node = graph.vertices[node].right
    neighbor = graph.vertices[node].child_node
    cost_of_finding_node = float('inf')
    node_graph_position = None
    if next_node != float('inf'):
        try:
            for i in neighbor:
                current_cost = graph.vertices[i].left + graph.vertices[node].child_node[i]
                if current_cost < cost_of_finding_node:
                    cost_of_finding_node = current_cost
                    node_graph_position = i
            if node_graph_position is None:
                print("Path search exhausted!")
            else:
                return node_graph_position
        except ValueError:
            print("Oops!  That was no valid move...  Path search exhausted.")

    else:
        print('You are stuck')


def maxQueue(queue):
    queue_size = len(queue)
    queue.sort()
    if queue_size > 0:
        return queue[0][:2]
    else:
        return float('inf'), float('inf')


def robot_sensor(graph, queue, current_state, scan_range, key):
    states_to_update = {}
    range_checked = 0
    if scan_range >= 1:
        for neighbor in graph.vertices[current_state].child_node:
            neighbor_coords = [int(neighbor.split('x')[1].split('y')[0]), int(neighbor.split('x')[1].split('y')[1])]
            states_to_update[neighbor] = graph.cells[neighbor_coords[1]
            ][neighbor_coords[0]]
        range_checked = 1

    while range_checked < scan_range:
        new_set = {}
        for state in states_to_update:
            new_set[state] = states_to_update[state]
            for neighbor in graph.vertices[state].child_node:
                if neighbor not in new_set:
                    neighbor_coords = [int(neighbor.split('x')[1].split('y')[0]), int(neighbor.split('x')[1].split('y')[1])]
                    new_set[neighbor] = graph.cells[neighbor_coords[1]
                    ][neighbor_coords[0]]
        range_checked += 1
        states_to_update = new_set

    new_obstacle = False
    for state in states_to_update:
        if states_to_update[state] < 0:
            for neighbor in graph.vertices[state].child_node:
                if graph.vertices[state].child_node[neighbor] != float('inf'):
                    neighbor_coords = [int(state.split('x')[1].split('y')[0]), int(state.split('x')[1].split('y')[1])]
                    graph.cells[neighbor_coords[1]][neighbor_coords[0]] = -2
                    graph.vertices[neighbor].child_node[state] = float('inf')
                    graph.vertices[state].child_node[neighbor] = float('inf')
                    update_vertex(graph, queue, state, key)
                    new_obstacle = True

    return new_obstacle


# for each element in the queue, map a key
def queue_map(graph, node, key):
    x_coordinate = abs(int(node.split('x')[1][0]) - int(node.split('x')[1][0]))
    y_coordinate = abs(int(node.split('y')[1][0]) - int(node.split('y')[1][0]))
    result = max(x_coordinate, y_coordinate)
    return (min(graph.vertices[node].left, graph.vertices[node].right) + result + key,
            min(graph.vertices[node].left, graph.vertices[node].right))


def update_vertex(graph, queue, node, key):
    goal_state = graph.goal_node
    if node != goal_state:
        min_node = float('inf')
        for i in graph.vertices[node].child_node:
            min_node = min(
                min_node, graph.vertices[i].left + graph.vertices[node].child_node[i])
        graph.vertices[node].right = min_node
    id_in_queue = [item for item in queue if node in item]
    if id_in_queue:
        try:
            if len(id_in_queue) != 1:
                print('multiple ' + node + ' can not be in the queue!')
        except ValueError:
            print('Remove multiple nodes ' + node + ' from queue!')
        queue.remove(id_in_queue[0])
    if graph.vertices[node].right != graph.vertices[node].left:
        heapq.heappush(queue, queue_map(graph, node, key) + (node,))


# Implement shortest path using Dijkstra's Algorithm
def dijkstra_shortest_path(graph, queue, current_node, key):
    queue.sort()
    queue_length = len(queue)
    if queue_length > 0:
        pop_result = queue[0][:2]
    else:
        pop_result = (float('inf'), float('inf'))
    queue_length = pop_result
    while (graph.vertices[current_node].right != graph.vertices[current_node].left) or (maxQueue(queue) < queue_map(graph, current_node, key)):
        k_old = maxQueue(queue)
        u = heapq.heappop(queue)[2]
        if k_old < queue_map(graph, u, key):
            heapq.heappush(queue, queue_map(graph, u, key) + (u,))
        elif graph.vertices[u].left > graph.vertices[u].right:
            graph.vertices[u].left = graph.vertices[u].right
            for i in graph.vertices[u].parent_node:
                update_vertex(graph, queue, i, key)
        else:
            graph.vertices[u].left = float('inf')
            update_vertex(graph, queue, u, key)
            for i in graph.vertices[u].parent_node:
                update_vertex(graph, queue, i, key)


class plot_robot_grid(Graph):
    def __init__(self, x_dim, y_dim, connect8=True):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        # First make an element for each row (height of grid)
        self.cells = [0] * y_dim
        # Go through each element and replace with row (width of grid)
        for i in range(y_dim):
            self.cells[i] = [0] * x_dim
        self.connect8 = connect8
        self.vertices = {}
        self.loadgraph()

    def loadgraph(self):
        edge = 1
        for i in range(len(self.cells)):
            row = self.cells[i]
            for j in range(len(row)):
                node = Node('x' + str(i) + 'y' + str(j))
                if i > 0:  # not top row
                    node.parent_node['x' + str(i - 1) + 'y' + str(j)] = edge
                    node.child_node['x' + str(i - 1) + 'y' + str(j)] = edge
                if i + 1 < self.y_dim:  # not bottom row
                    node.parent_node['x' + str(i + 1) + 'y' + str(j)] = edge
                    node.child_node['x' + str(i + 1) + 'y' + str(j)] = edge
                if j > 0:  # not left col
                    node.parent_node['x' + str(i) + 'y' + str(j - 1)] = edge
                    node.child_node['x' + str(i) + 'y' + str(j - 1)] = edge
                if j + 1 < self.x_dim:  # not right col
                    node.parent_node['x' + str(i) + 'y' + str(j + 1)] = edge
                    node.child_node['x' + str(i) + 'y' + str(j + 1)] = edge
                self.vertices['x' + str(i) + 'y' + str(j)] = node


# Define colors
black = (0, 0, 0)
white = (255, 255, 255)
green = (0, 255, 0)
red = (255, 0, 0)
grey1 = (145, 145, 102)
grey2 = (77, 77, 51)
blue = (0, 0, 80)
grey3 = (132, 75, 30)

colors = {
    0: grey3,
    1: green,
    -1: red,
    -2: red
}

# cell dimensions
width = 40
height = 40

margin = 5

grid = []
for row in range(10):
    grid.append([])
    for column in range(10):
        grid[row].append(0)

pygame.init()

x_dimension = 12
y_dimension = 12
viewing_range = 24

window_size = [(width + margin) * x_dimension + margin, (height + margin) * y_dimension + margin]
screen = pygame.display.set_mode(window_size)

pygame.display.set_caption("Robotic path-planning algorithm")

# Loop until the user clicks the close button
done = False

clock = pygame.time.Clock()

if __name__ == "__main__":

    graph = plot_robot_grid(x_dimension, y_dimension)
    start_state = 'x0y0'
    goals = ['x0y6','x0y7','x0y8','x0y9','x0y10','x0y11', 'x1y6', 'x1y7','x1y8','x1y9','x1y10','x1y11', 'x2y6','x2y7',
             'x2y8','x2y9','x2y10','x2y11',
             'x3y6','x3y7','x3y8','x3y9','x3y10','x3y11', 'x4y6','x4y7','x4y8','x4y9','x4y10','x4y11', 'x5y6','x5y7',
             'x5y8','x5y9','x5y10','x5y11',
             'x6y6','x6y7','x6y8','x6y9','x6y10','x6y11', 'x7y6','x7y7','x7y8','x7y9','x7y10','x7y11', 'x8y6','x8y7',
             'x8y8','x8y9','x8y10','x8y11',
             'x9y6','x9y7','x9y8','x9y9','x9y10','x9y11', 'x10y6','x10y7','x10y8','x10y9','x10y10','x10y11', 'x11y6',
             'x11y7','x11y8','x11y9','x11y10','x6y11']
    goal_state = random.choice(goals)
    goal_coords = [int(goal_state.split('x')[1].split('y')[0]), int(goal_state.split('x')[1].split('y')[1])]

    graph.set_start_node(start_state)
    graph.set_goal_node(goal_state)
    key = 0
    last_state = start_state
    queue = []

    graph.vertices[goal_state].right = 0
    heapq.heappush(queue, queue_map(graph, goal_state, key) + (goal_state,))
    dijkstra_shortest_path(graph, queue, start_state, key)

    current_state = start_state
    pos_coords = [int(current_state.split('x')[1].split('y')[0]), int(current_state.split('x')[1].split('y')[1])]

    basicfont = pygame.font.SysFont('Comic Sans MS', 36)

    while not done:
        for event in pygame.event.get():  # User did something
            if event.type == pygame.QUIT:  # If user clicked close
                done = True  # Flag that we are done so we exit this loop
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:

                if current_state == graph.goal_node:
                    new_state = 'goal'
                    key = key
                else:
                    last_state = current_state
                    new_state = compute_next_node(graph, current_state)
                    new_coords = [int(new_state.split('x')[1].split('y')[0]), int(new_state.split('x')[1].split('y')[1])]

                    if graph.cells[new_coords[1]][new_coords[0]] == -1:
                        new_state = current_state

                    results = robot_sensor(graph, queue, new_state, viewing_range, key)
                    x_coordinate = abs(int(last_state.split('x')[1][0]) - int(last_state.split('x')[1][0]))
                    y_coordinate = abs(int(last_state.split('y')[1][0]) - int(last_state.split('y')[1][0]))
                    max_result = max(x_coordinate, y_coordinate)
                    key += max_result
                    dijkstra_shortest_path(graph, queue, current_state, key)

                if new_state == 'goal':
                    print('Goal Reached!')
                    done = True
                else:
                    current_state = new_state
                    pos_coords = [int(current_state.split('x')[1].split('y')[0]), int(current_state.split('x')[1].split('y')[1])]

            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                column = pos[0] // (width + margin)
                row = pos[1] // (height + margin)
                if(graph.cells[row][column] == 0):
                    graph.cells[row][column] = -1

        for row in range(y_dimension):
            for column in range(x_dimension):
                color = white
                pygame.draw.rect(screen, colors[graph.cells[row][column]],
                                 [(margin + width) * column + margin,
                                  (margin + height) * row + margin, width, height])
                node_name = 'x' + str(column) + 'y' + str(row)

                if(graph.vertices[node_name].left != float('inf')):
                    text = basicfont.render(
                        str(graph.vertices[node_name].left), True, (0, 0, 200))
                    textrect = text.get_rect()
                    textrect.centerx = int(
                        column * (width + margin) + width / 2) + margin
                    textrect.centery = int(
                        row * (height + margin) + height / 2) + margin
                    image = pygame.image.load('background.jpeg')
                    if (graph.cells[row][column] == -1):
                        image = pygame.image.load('background1.png')

                    screen.blit(image, textrect)

        pygame.draw.rect(screen, green, [(margin + width) * goal_coords[0] + margin,
                                         (margin + height) * goal_coords[1] + margin, width, height])

        robot_center = [int(pos_coords[0] * (width + margin) + width / 2) +
                        margin, int(pos_coords[1] * (height + margin) + height / 2) + margin]
        pygame.draw.circle(screen, red, robot_center, int(width / 2) - 2)

        pygame.draw.rect(
            screen, blue, [robot_center[0] - viewing_range * (width + margin),
                           robot_center[1] - viewing_range * (height + margin),
                           2 * viewing_range * (width + margin),
                           2 * viewing_range * (height + margin)], 2)

        # Limit to 60 frames per second
        clock.tick(20)

        pygame.display.flip()

    pygame.quit()

