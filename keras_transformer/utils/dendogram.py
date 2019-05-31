# TODO (fabawi): this is currently an experimental draft. DO NOT USE!



from nltk import Tree
from nltk.draw.util import CanvasFrame
from nltk.draw import TreeWidget

def recursive_len(item):
    if type(item) == Tree or type(item) == list:
        return sum(recursive_len(subitem) for subitem in item)
    else:
        return 1

def traverse_tree(tree, level):
    levels = list()
    levels.extend([level])
    for subtree in tree:
        if type(subtree) == Tree:
            levels.extend(traverse_tree(subtree, level/1.5))
        else:
            levels.extend([level])

    return levels

# print(len("( event : ( action : move ( token : 1 ) ) ( entity : ( color : blue ( token : 3 ) ) ( type : cube ( token : 4 ) ) ) ( destination : ( spatial-relation : ( relation : above ( token : 5 7 ) ) ( entity : ( color : blue ( token : 9 ) ) ( type : cube ( token : 10 ) ) ) ) ) )".split(' ')))
cf = CanvasFrame()
t = Tree.fromstring("(sequence: (event: (action: take (token: 1)) (entity: (id: 1) (color: blue (token: 3)) (type: cube (token: 4)) (spatial-relation: (relation: above (token: 7 9)) (entity: (color: yellow (token: 10)) (type: cube (token: 11)))))) (event: (action: drop (token: 13)) (entity: (type: reference (token: 14)) (reference-id: 1)) (destination: (spatial-relation: (relation: above (token: 15 17)) (entity: (color: red (token: 18)) (type: cube (token: 19)))))))".replace(':',''))
t.pretty_print()
tc = TreeWidget(cf.canvas(),t)
cf.add_widget(tc,10,10) # (10,10) offsets
cf.print_to_file('tree.ps')

# print(repr(t))
# print(recursive_len(t))
levels = traverse_tree(t,15)


import plotly.offline as py
import plotly.figure_factory as ff

import numpy as np

dend_matrix = None
for level in levels:
    dend = np.ones([1, 47]) * level
    if dend_matrix is None:
        dend_matrix = dend
    else:
        dend_matrix = np.concatenate([dend_matrix,dend])

names = "( event : ( action : move ( token : 1 ) ) ( entity : ( color : blue ( token : 3 ) ) ( type : cube ( token : 4 ) ) ) ( destination : ( spatial-relation : ( relation : above ( token : 5 7 ) ) ( entity : ( color : blue ( token : 9 ) ) ( type : cube ( token : 10 ) ) ) ) ) )".replace(' :', '').replace('(', '').replace(')','').split(' ')
fig = ff.create_dendrogram(dend_matrix, orientation='left', labels=names)
fig['layout'].update({'width':800, 'height':800})
py.plot(fig)