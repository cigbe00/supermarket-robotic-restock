########################################################
#                      Node.py                         #
#                                                      #
#     Created by Ngwudike & Chukwudi on 11/20/2020.    #
#                                                      #
########################################################


class Node:
    def __init__(self, node):
        self.node = node
        self.parent_node = {}
        self.child_node = {}
        self.left = float('inf')
        self.right = float('inf')

    def __str__(self):
        return 'Node: ' + self.node + ' left: ' + str(self.left) + ' right: ' + str(self.right)

    def __repr__(self):
        return self.__str__()

    def set_parent_node(self, parents):
        self.parent_node = parents

    # getters and setters
    def set_child_node(self, child_node):
        self.child_node = child_node

    def get_child_node(self):
        return self.child_node

    def get_parent_node(self):
        return self.parent_node



