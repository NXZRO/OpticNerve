from kdquery import Tree
import numpy as np
import time

def print_tree(tree,cap):
    for id in range(cap):
        node = tree.get_node(id)
        print(id,node)

k = 2
cap = 10
l = [-5, 5]

region = [l for _ in range(k)]

tree = Tree(k, cap, region)

p0 = np.array([3,0])
p1 = np.array([2,0])
p2 = np.array([-2,0])
p3 = np.array([1,0])

ps = [p0,p1,p2,p3]
for i,p in enumerate(ps):
    tree.insert(p,i)
    print_tree(tree, cap)
    print("-----------------------------------------------------------------------------------------------------------")


# k = 128
# cap = 1000
# l = [-1, 1]
#
#
# region = [l for _ in range(k)]
#
# tree = Tree(k, cap, region)
#
# target = np.random.rand(1, k)
#
# raw_dataset = np.random.rand((1000-100), k)
#
# dataset = np.append(raw_dataset, target, axis=0)
#
# np.random.shuffle(dataset)
#
# print("build ...")
# print("cap: ", cap)
# start = time.clock()
#
# for data in dataset:
#     tree.insert(data, data)
#
# end = time.clock()
# print("building cost: {} sec".format(end-start))
#
#
# print("search ...")
# start = time.clock()
# node_id, dist = tree.find_nearest_point(target[0])
# end = time.clock()
# print("searing cost: {} sec".format(end-start))
#
# print("nide id:", node_id)
# print("dist:", dist)
# node = tree.get_node(node_id)
# print("node data:", node.data)


