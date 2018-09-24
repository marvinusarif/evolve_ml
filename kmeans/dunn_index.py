import sys


def cluster_membership(predicted_centroids):
    data_membership={}
    for i in range(len(predicted_centroids)):
        p_class=predicted_centroids[i]
        if p_class in data_membership:
            data_membership[p_class].append(i)
        else:
            data_membership[p_class]=[i]

    print('============membership============', data_membership)

    return data_membership


def min_distance_between_centroids(distance_func, centroids):
    min_d=sys.float_info.max
    for i in range(len(centroids)-1):
        for j in range(i+1, len(centroids)):
            d=distance_func(centroids[i], centroids[j])
            if d<min_d:
                min_d=d

    return min_d


def max_distance_to_centroids(distance_func, data, centroids, membership_data):
    max_d=sys.float_info.min
    for k,vs in membership_data.items():
        for v in vs:
            d=distance_func(centroids[k], data[v])
            if d>max_d:
                max_d=d

    return max_d


def compute_dunn_index(distance_func, predicted_centroids, data, centroids):
    data_membership=cluster_membership(predicted_centroids)
    min_d_centroids=min_distance_between_centroids(distance_func, centroids)
    max_d_within_centroids=max_distance_to_centroids(distance_func, data,centroids,data_membership)
    di=min_d_centroids/max_d_within_centroids
    return di
