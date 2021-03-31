########################################################
#                      Graph.py                        #
#                                                      #
#     Created by Ngwudike & Chukwudi on 11/20/2020.    #
#                                                      #
########################################################


class Graph:
    def __init__(self):
        #self.y_axis = None
        #self.x_axis = None
        self.vertices = {}
        self.start_node = None
        self.goal_node = None
        #self.cells = None

    def set_start_node(self, node):
        try:
            if self.vertices[node]:
                self.start_node = node
            else:
                print("Invalid node provided")
        except ValueError:
            print("The node provided can not be found on the vertices for this graph")

    def set_goal_node(self, node):
        try:
            if self.vertices[node]:
                self.goal_node = node
            else:
                print("Invalid node provided")
        except ValueError:
            print("The node provided can not be found on the vertices for this graph")
