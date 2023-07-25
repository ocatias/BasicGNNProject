# Python program to find bridges in a given undirected graph
# Complexity : O(V+E)
# Adapted from https://www.geeksforgeeks.org/biconnected-components
from collections import defaultdict
from copy import deepcopy


# This class represents an undirected graph using adjacency list representation
class BiconnectedComponents:

    def __init__(self, graph):
        self.V = graph['num_nodes']  # No. of vertices
        self.graph = defaultdict(list)  # default dictionary to store graph
        self.Time = 0
        # Count is number of biconnected components
        self.count = 0
        for i in range(graph.edge_attr.shape[0]):
            self.addEdge(int(graph.edge_index[0][i]), int(graph.edge_index[1][i]))
        self.groups = []

    # function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)

    '''A recursive function that finds and prints strongly connected
    components using DFS traversal
    u --> The vertex to be visited next
    disc[] --> Stores discovery times of visited vertices
    low[] -- >> earliest visited vertex (the vertex with minimum
               discovery time) that can be reached from subtree
               rooted with current vertex
    st -- >> To store visited edges'''

    def BCCUtil(self, u, parent, low, disc, st):

        # Count of children in current node
        children = 0

        # Initialize discovery time and low value
        disc[u] = self.Time
        low[u] = self.Time
        self.Time += 1

        # Recur for all the vertices adjacent to this vertex
        for v in self.graph[u]:
            # If v is not visited yet, then make it a child of u
            # in DFS tree and recur for it
            if disc[v] == -1:
                parent[v] = u
                children += 1
                st.append((u, v))  # store the edge in stack
                self.BCCUtil(v, parent, low, disc, st)

                # Check if the subtree rooted with v has a connection to
                # one of the ancestors of u
                # Case 1 -- per Strongly Connected Components Article
                low[u] = min(low[u], low[v])

                # If u is an articulation point, pop
                # all edges from stack till (u, v)
                if parent[u] == -1 and children > 1 or parent[u] != -1 and low[v] >= disc[u]:
                    self.count += 1  # increment count
                    w = -1
                    group_vertices = set()
                    while w != (u, v):
                        w = st.pop()
                        group_vertices.add(w[0])
                        group_vertices.add(w[1])
                    self.groups.append(deepcopy(group_vertices))

            elif v != parent[u] and low[u] > disc[v]:
                '''Update low value of 'u' only of 'v' is still in stack
                (i.e. it's a back edge, not cross edge).
                Case 2
                -- per Strongly Connected Components Article'''

                low[u] = min(low[u], disc[v])

                st.append((u, v))

    # The function to do DFS traversal.
    # It uses recursive BCCUtil()
    def BCC(self):
        # Initialize disc and low, and parent arrays
        disc = [-1] * (self.V)
        low = [-1] * (self.V)
        parent = [-1] * (self.V)
        st = []

        # Call the recursive helper function to
        # find articulation points
        # in DFS tree rooted with vertex 'i'
        for i in range(self.V):
            if disc[i] == -1:
                self.BCCUtil(i, parent, low, disc, st)

            # If stack is not empty, pop all edges from stack
            if st:
                self.count = self.count + 1
                group_vertices = set()

                while st:
                    w = st.pop()
                    group_vertices.add(w[0])
                    group_vertices.add(w[1])
                self.groups.append(deepcopy(group_vertices))

        return [list(x) for x in self.groups if len(x) > 2]
