from anytree import Node, RenderTree, PreOrderIter, AsciiStyle
from anytree.exporter import DotExporter
#use anytree instead, it has nice prints / plots
#https://pypi.org/project/anytree/

class Tree():
    def __init__(self, data, parent):
        self.data = data
        self.parent = parent
        self.children = []
    def __eq__(self, other):
        if isinstance(other, Tree):
            return self.data == other.data
        else:
            return False
    def __repr__(self):
        return "Tree("+str(self.data)+","+str(self.children)+")"
    def __str__(self):
        return self.__repr__()
    def update_parent(self, new):
        self.parent = new
    def add_child(self, c):
        self.children.append(c)
    def rm_child(self, c):
        self.children.remove(c)

#sl subtemplate of bl (matching to the same or a subset of patterns)
def sub(sl, bl):
    # Wenn gleich True
    if sl==bl: return True
    if len(bl)==0: return False
    if len(sl)==0:
        b, *bs = bl
        if b == "*":
            return sub(sl, bs)
        return False
    # 1. Element s/b || Rest ss/bs
    s, *ss = sl
    b, *bs = bl
    
    if b=="*":
        if s == "*":
            return sub(ss, bs) or sub(ss, bl)
        else:
            return sub(ss, bs) or sub(ss, bl) or sub(sl, bs)
    elif b=="_":
        if s=="*": return False
        else: return sub(ss, bs)
    else:
        if s == "*" or s == "_": return False
        elif s == b: return sub(ss, bs)
        else: return False

#get the parent candidate for a template, i.e. a node with a template that matches a superset but no child matches a superset
def get_parent_can(tree, template):
    if tree.data == template:
        return tree
    if sub(template, tree.data):
        if len(tree.children)==0:
            return tree
        tmp = [get_parent_can(c, template) for c in tree.children]
        tmp = [x for x in tmp if x!=False]
        if len(tmp)==0:
            return tree
        elif len(tmp)==1:
            return tmp[0]
        else:
            raise ValueError('more than one child can be parent of '+str(template)+"   "+str(tree.children))
    else:
        return False
    
#insert templeate into the tree:
#find parent candidate
#if it has the same template noting changes
#otherwise check wether some children need to be reorganized and insert the new node properly
def update_tree(tree, template):
    new_parent = get_parent_can(tree, template)
    if new_parent.data == template:
        return tree
    tmp = [sub(c.data, template) for c in new_parent.children]
    if any(tmp):
        rm = []
        new_node = Tree(template, new_parent)
        for i in range(len(tmp)):
            if tmp[i]:
                moved = new_parent.children[i]
                rm.append(moved)
                new_node.add_child(moved)
                moved.update_parent(new_node)
        for i in rm:
            new_parent.rm_child(i)
        new_parent.add_child(new_node)
    else:
        new_parent.add_child(Tree(template, new_parent))

def t2anytree(tree):
    if len(tree.children)>0:
        ch_nodes = [t2anytree(x) for x in tree.children]
        return Node(str(tree.data), children=ch_nodes)
    else:
        return Node(str(tree.data))