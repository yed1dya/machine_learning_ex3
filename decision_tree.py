import itertools
import math


class Node:
    def __init__(self, points: list[(float, float, int)] = None, split_feature: int = 0,
                 split_value: float = float('inf'), leaf: bool = False, label: int = 0):
        if points is None:
            points = []
        self.points = points
        self.split_feature = split_feature
        self.split_value = split_value
        self.leaf = leaf
        self.label = label
        self.error = 0
        self.entropy = 0
        self.left, self.right = None, None

    def __repr__(self):
        return str(self)

    def __str__(self):
        if self.leaf:
            if not self.points:
                return f"Leaf(label={self.label}, points=None)"
            return f"Leaf(label={self.label}, points={len(self.points)}, error={self.error})"
        if not self.points:
            return f"Node(split_feature={self.split_feature}, split_value={self.split_value}, points=None)"
        return (f"Node(split_feature={self.split_feature}, split_value={self.split_value}, "
                f"points={len(self.points)})")

    def calc_entropy(self):
        n = len(self.points)
        a = sum(1 for p in self.points if p[2] == 0)
        b = n - a
        if n < 1 or a == 0 or b == 0:
            return 0
        self.entropy = ((a / n) * math.log2(n / a)) + ((b / n) * math.log2(n / b))
        return self.entropy

    def vote(self):
        count = [0, 0]
        for p in self.points:
            count[p[2]] += 1
        self.label = 0 if count[0] > count[1] else 1
        return self.label

    def calc_error(self):
        self.error = sum(1 for p in self.points if p[2] != self.label)
        return self.error

    def split(self, left, right):
        self.left, self.right = left, right
        for p in self.points:
            if p[self.split_feature] <= self.split_value:
                left.points.append(p)
            else:
                right.points.append(p)


def draw(node: Node = None, level: int = 0):
    if node:
        print(f"{"    " * level}{str(node)}")
        draw(node.left, level + 1)
        draw(node.right, level + 1)


# parse data from 'iris.txt' to create the lists of points with labels, and lists of x,y values
def make_points_list(file_name: str, class1: str, class2: str) -> \
        (list[(float, float, int)], list[float], list[float]):
    with open(file_name, 'r') as file:
        x_values, y_values, points = [], [], []
        for line in file:
            t = line.split()
            x, y = float(t[1]), float(t[2])
            if t[4] in {class1, class2}:
                points.append((x, y, 1 if t[4] == class1 else 0))
            if x not in x_values:
                x_values.append(x)
            if y not in y_values:
                y_values.append(y)
    return points, sorted(x_values), sorted(y_values)


def problem2a(points, split_params):
    print("problem 2a: brute force")
    n = len(split_params)
    print(f"searching with {n} possible params")
    best_error, best_perm, root = float('inf'), (None, -1), None
    count, total = 0, (n * (n - 1) * (n - 2))
    for perm in itertools.permutations(range(n), 3):  # Unique 3-node combinations
        count += 1
        if count % 10000 == 0 or count == 1:
            print(f"perm {count} / {total}: {perm}")
        a, b, c = perm
        node_a = Node(points=points, split_feature=split_params[a][0], split_value=split_params[a][1])
        node_b = Node(split_feature=split_params[b][0], split_value=split_params[b][1])
        node_c = Node(split_feature=split_params[c][0], split_value=split_params[c][1])
        node_d, node_e, node_f, node_g = Node(leaf=True), Node(leaf=True), Node(leaf=True), Node(leaf=True)

        # split nodes
        node_a.split(node_b, node_c)
        node_b.split(node_d, node_e)
        node_c.split(node_f, node_g)

        # leaves
        error = 0
        for n in [node_d, node_e, node_f, node_g]:
            n.vote()
            error += n.calc_error()

        # Check if this tree is better
        if error < best_error:
            root, left, right, leaf_ll, leaf_lr, leaf_rl, leaf_rr = (
                node_a, node_b, node_c, node_d, node_e, node_f, node_g)
            best_error = error
            best_perm = perm, count
            print(f"best error = {error}")
            if error == 0:
                break
    print(f"\nbest tree with error = {best_error} found with ({best_perm[0]}) at try {best_perm[1]}/{total}")
    draw(root)


def find_best(split_params, points):
    best_entropy = float('inf')
    root = left = right = None
    for feature, value in split_params:
        node_a = Node(points=points, split_feature=feature, split_value=value)
        node_b, node_c = Node(), Node()
        node_a.split(node_b, node_c)
        entropy = node_b.calc_entropy() + node_c.calc_entropy()
        if entropy < best_entropy:
            best_entropy = entropy
            root, left, right = node_a, node_b, node_c
    root.calc_entropy()
    return root, left, right


def problem2b(points, split_params):
    print("problem 2b: entropy")
    print("finding best split params for root...")
    root, left, right = find_best(split_params, points)
    print("finding best split params for left...")
    left, leaf_ll, leaf_lr = find_best(split_params, left.points)
    print("finding best split params for right...")
    right, leaf_rl, leaf_rr = find_best(split_params, right.points)
    leaf_ll.leaf = leaf_lr.leaf = leaf_rl.leaf = leaf_rr.leaf = True
    left.left, left.right, right.left, right.right = leaf_ll, leaf_lr, leaf_rl, leaf_rr
    root.left, root.right = left, right
    error = 0
    print("setting leaf decision labels")
    for n, s in [(leaf_ll, "LL"), (leaf_lr, "LR"), (leaf_rl, "RL"), (leaf_rr, "RR")]:
        n.vote()
        error += n.calc_error()
    print("\nbest tree with error:", error)
    draw(root)


def main():
    points, x_vals, y_vals = make_points_list("iris.txt", 'Iris-versicolor', 'Iris-virginica')
    split_params = [(0, x) for x in x_vals] + [(1, y) for y in y_vals]
    problem2a(points, split_params)
    print("\n----------------------------------------\n")
    problem2b(points, split_params)


if __name__ == '__main__':
    main()
