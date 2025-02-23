# 207199282, 207404997
import numpy as np


# parse data from 'iris.txt' to create the lists of points with labels
def make_points_list(file_name: str, class1: str, class2: str, x: int, y: int):
    points = []
    with open(file_name, 'r') as file:
        for line in file:
            t = line.split()
            if t[4] == class1:
                points.append([float(t[x]), float(t[y]), -1])
            elif t[4] == class2:
                points.append([float(t[x]), float(t[y]), 1])
    return points


# l_p distance function
def l_dist(p, point, other):
    x_dist, y_dist = abs(point[0] - other[0]), abs(point[1] - other[1])
    if p == float('inf'):
        return max([x_dist, y_dist])
    x_dist, y_dist = x_dist ** p, y_dist ** p
    return (x_dist + y_dist) ** (1 / p)


# calculate error rate (empirical or true)
def calc_error(p, k, base, points):
    error = 0
    for point in points:
        distances = [l_dist(p, point, z) for z in base]
        sorted_indexes = np.argsort(distances)[:k]
        nearest_labels = [base[i][2] for i in sorted_indexes]
        error += np.sign(sum(nearest_labels)) != point[2]
    return error / len(points)


# main k-nn algorithm
def knn(p: int, k: int, train: list[dict], test: list[dict]):
    return calc_error(p, k, train, train), calc_error(p, k, train, test)  # empirical and true errors


# driver code for the assignment
def problem1(k_list, p_list, points, runs):
    print(f"p    k     avg emp       avg true        avg diff\n")
    true_results = np.zeros((len(p_list), len(k_list)))
    min_diff = min_emp = min_true = [float('inf'), -1, -1]
    for i in range(len(p_list)):  # for all options of p
        for j in range(len(k_list)):  # for all options of k
            emp_errors, true_errors, p, k = [], [], p_list[i], k_list[j]
            for run in range(runs):  # get average over 100 runs of knn
                train, test = [], []
                for point in points:
                    if np.random.random() < 0.5:
                        train.append(point)
                    else:
                        test.append(point)
                e, t = knn(p, k, train, test)  # get errors from k-nn algorithm
                emp_errors.append(e)
                true_errors.append(t)
            avg_emp, avg_true = float(np.mean(emp_errors)), float(np.mean(true_errors))
            # outputs:
            s = str(p) if p == float('inf') else f"{p}  "
            s += f"  {k}      {avg_emp:.4f}         {avg_true:.4f}          {(avg_true - avg_emp):.4f}"
            print(s)
            true_results[i][j] = avg_true
            min_emp = [avg_emp, p, k] if avg_emp < min_emp[0] else min_emp
            min_true = [avg_true, p, k] if avg_true < min_true[0] else min_true
            min_diff = ([abs(avg_emp - avg_true), p, k]
                        if abs(avg_emp - avg_true) < min_diff[0] else min_diff)
        print()
    print(f"min true error: {min_true[0]:.4f} with p={min_true[1]}, k={min_true[2]}")


def main():
    class1, class2, x, y = 'Iris-versicolor', 'Iris-virginica', 1, 2
    runs = 100
    k_list = [1, 3, 5, 7, 9]
    p_list = [1, 2, float('inf')]
    points = make_points_list('iris.txt', class1, class2, x, y)
    problem1(k_list, p_list, points, runs)


if __name__ == '__main__':
    main()
