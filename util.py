from dataset import Dales
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


def fps(data):
    pnts = []
    pnts.append(data[0])
    while len(pnts) != num_points:
        pnts.append(cal_mx_distance(pnts[-1], data))

    return pnts

if __name__ == "__main__":
    # train = Dales(25, 2048)
    num_points = 10
    # train = fps(train[0][0])
    # print(len(train))
    # # print(train[0][0][0])

    # print(train)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(train)
    torch.manual_seed(42)

    a = torch.randint(10, (5,1))
    b = torch.randint(10, (5,1))

    c = np.minimum(a,b)
    print(a, b, c)
    # o3d.visualization.draw_geometries([pcd])