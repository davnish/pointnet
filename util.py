from dataset import Dales, modelnet40
import open3d as o3d
import numpy as np
import torch
torch.manual_seed(42)


def cal_mx_distance(pnt, data):
    dist = []
    euclidean_dist = torch.sum((data - pnt)**2)
    dist.append(euclidean_dist)
    
    dist = torch.tensor(dist)
    return data[torch.argmax(dist)]


def fps(points, num_points):
    points = np.asarray(points)
    sampled_pnts = np.zeros(num_points, dtype='int') # Sampled points
    pnts_left = np.arange(len(points), dtype='int') # points not sampled 
    dist = np.ones_like(pnts_left) * float('inf') # dist array

    selected = 0 # Selected current point
    sampled_pnts[0] = selected
    pnts_left = np.delete(pnts_left, selected)
    # dist = np.linalg.norm(points[pnts_left] - points[selected], ord = 2)


    for i in range(1, num_points):
        

        selected_dist = np.linalg.norm(points[pnts_left] - points[selected], ord = 2)

        # temp = np.linalg.norm(points[pnts_left] - points[selected], ord = 2)

        dist[pnts_left] = np.minimum(dist[pnts_left], selected_dist)
        # print(dist)
        selected = np.argmax(dist[pnts_left], axis = 0)
        # print(selected)
        sampled_pnts[i] = pnts_left[selected]

        pnts_left = np.delete(pnts_left, selected)

    return points[sampled_pnts]


if __name__ == "__main__":
    train = modelnet40()
    num_points = 10
    print(train[0][0])
    # train = fps(train[0][0], 2048)
    train = train[0][0]
    print(train)
    # a = np.arange(10)
    # print(a)
    # a = np.delete(a, 2)
    # print(a)
    # train = fps(train[0][0])
    # print(len(train))
    # # print(train[0][0][0])

    # print(train)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(train)
    # torch.manual_seed(42)

    # a = torch.randint(10, (5,1))
    # b = torch.randint(10, (5,1))

    # c = np.minimum(a,b)
    # print(a, b, c)
    o3d.visualization.draw_geometries([pcd])