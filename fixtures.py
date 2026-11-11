"""Literal transcription of PDF pages 4, 7, 10-12 and 14."""
from search_algorithms import TreeNode

MAZE = [list("S..X."), list(".X..."), list(".X.X."), list("...XE"), list("XX...")]
MAZE_START, MAZE_GOAL = (0, 0), (3, 4)
WEIGHTED = [["S", 1, 1, 8, 1], [1, 8, 3, 1, 1], [1, 8, 1, 8, 1],
            [1, 1, 1, 8, 1], [8, 8, 1, 1, "E"]]
WEIGHTED_START, WEIGHTED_GOAL = (0, 0), (4, 4)
TREASURE_GRID = [
    [1, 1, 1, None, 2, 1, 1],
    [1, None, 6, None, 1, None, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [None, 1, None, 8, None, 1, None],
    [1, 1, 1, 1, 1, 1, 1],
    [1, None, 9, None, 1, None, 1],
    [1, 1, 1, None, 1, 1, 1],
]
NAMES = ["A", "B", "T1", "T2", "T3", "T4"]
POSITIONS = [(0, 0), (6, 6), (0, 6), (2, 2), (4, 4), (6, 0)]
VALUES = {2: 9, 3: 6, 4: 7, 5: 7}
EXPECTED_DISTANCES = [[0,12,10,4,8,8], [12,0,8,8,4,10], [10,8,0,6,6,12],
                      [4,8,6,0,6,6], [8,4,6,6,0,6], [8,10,12,6,6,0]]


def example_tree():
    # PDF's code reuses max2. Searches traverse each occurrence without
    # mutating the shared object, so this matches the expanded JSON tree.
    max1 = TreeNode(children=[TreeNode(3), TreeNode(5)])
    max2 = TreeNode(children=[TreeNode(6), TreeNode(9)])
    max3 = TreeNode(children=[TreeNode(1), TreeNode(2)])
    return TreeNode(children=[TreeNode(children=[max1, max2]),
                              TreeNode(children=[max2, max3])])
