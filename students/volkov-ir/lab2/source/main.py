import numpy as np
from data_preprocessing import prepare_data

# X уже стандартизованы, y — номера видов 0/1/2
X_train, X_test, y_train, y_test, feature_names, class_names = prepare_data()


def calculate_knn(X, y, u, k):
    #эвклидово расстояние
    square = np.square(u-X)
    sum_square = np.sum(square, axis=1)
    dist_ux = np.sqrt(sum_square)
    nearest_idx = np.argsort(dist_ux)
    shear = nearest_idx[0:k]
    shear_y = y[shear]
    classes, counts = np.unique(shear_y, return_counts=True)
    index_max = np.argmax(counts)
    return classes[index_max]


def calculate_parzen(X, y, u, k):
    square = np.square(u - X)
    sum_square = np.sum(square, axis=1)
    dist_ux = np.sqrt(sum_square)
    nearest_idx = np.argsort(dist_ux)
    h = dist_ux[nearest_idx[k]]
    shear = nearest_idx[0:k]
    shear_y = y[shear]
    p = dist_ux[shear]
    w = np.exp(-2 * (p / h) ** 2)
    sums_per_class = np.bincount(shear_y, weights=w, minlength=(y.max() + 1))
    pred = np.argmax(sums_per_class)
    return pred, sums_per_class

def calculate_parzen_lecture(X, y, u, k):
    square = np.square(u - X)
    sum_square = np.sum(square, axis=1)
    dist_ux = np.sqrt(sum_square)
    nearest_idx = np.argsort(dist_ux)
    h = dist_ux[nearest_idx[k]]
    p = dist_ux
    w = np.exp(-2 * (p / h) ** 2)
    sums_per_class = np.bincount(y, weights=w, minlength=(y.max() + 1))
    pred = np.argmax(sums_per_class)
    return pred, sums_per_class

def calculate_loo(X, y, k_values, method):
    loss = []
    for k in k_values:
        count_loss = 0
        for i in range(0, len(X)):
            X_k = X.copy()
            y_k = y.copy()
            u = X_k[i]
            y_i = y_k[i]
            X_k = np.delete(X_k, i , 0)
            y_k = np.delete(y_k, i, 0)
            if method == "knn":
                pred = calculate_knn(X_k, y_k, u, k)
            elif method == "parzen":
                pred, _ = calculate_parzen(X_k, y_k, u, k)
            elif method == "parzen_lecture":
                pred, _ = calculate_parzen_lecture(X_k, y_k, u, k)
            else:
                raise ValueError("method должен быть 'knn', 'parzen' или 'parzen_lecture'")
            if pred != y_i:
                count_loss += 1
        loss.append(count_loss/len(X))
    return loss


X = np.array([[1.0, 1.2], [2.3, 0.9], [1.4, 2.1], [2.6, 2.8],
              [6.1, 5.3], [5.7, 6.4], [7.2, 5.1]])
y = np.array([0, 0, 1, 1, 1, 1, 0])
k_values = [4, 1, 2]



def calculate_margin(X, y, k):
    margin = []
    for i in range(0, len(X)):
        X_k = X.copy()
        y_k = y.copy()
        u = X_k[i]
        y_i = y_k[i]
        X_k = np.delete(X_k, i, 0)
        y_k = np.delete(y_k, i, 0)
        pred, sums = calculate_parzen(X_k, y_k, u, k)
        new_sums = np.delete(sums, y_i, 0)
        margin.append(sums[y_i]-np.max(new_sums))
    return np.array(margin)


def calculate_start_omega(margin, y):
    indexes = []
    for c in np.unique(y):
        ind = np.flatnonzero(y == c)
        best_ind = np.argmax(margin[ind])
        indexes.append(ind[best_ind])
    return np.array(indexes)


def calculate_ccv(X, y, omega_ind):
    err = 0
    for i in range(0, len(X)):
        rest = omega_ind[omega_ind != i]
        res = calculate_knn(X[rest], y[rest], X[i], 1)
        if (res != y[i]):
            err += 1
    return err / len(X)


def greedy_addition(X, y, omega_ind):
    history = []
    res = calculate_ccv(X, y, omega_ind)
    for i in range(len(X)):
        candidates_cvv = []
        other_ind = np.setdiff1d(np.arange(len(X)), omega_ind)
        for ind in other_ind:
            candidate = np.append(omega_ind, ind)
            res_new = calculate_ccv(X, y, candidate)
            candidates_cvv.append(res_new)
                # if (res > np.argmin(candidate)):
                #     res = np.argmin(candidates_cvv)
                #     history.append(res)
                # else:
                #     break








    return omega_ind, history



margin = calculate_margin(X, y, 4)
omega_ind = calculate_start_omega(margin, y)
print(np.arange(len(X)))
print(omega_ind)
omega = calculate_ccv(X, y, omega_ind)
print(omega)