import numpy as np
import open3d as o3d
import laspy 
import os 
import torch
from model_pct import PointTransformerSeg
from dataset import Dales
from torch.utils.data import DataLoader
from main_seg import test_loop
np.random.seed(100)
# a = np.random.randint(0,255, (1,3)).repeat(2048, axis = 0)
# print(a.shape)
# print(a)



# vis2 = o3d.visualization.Visualizer()

# vis2.add_geometry(o3d.geometry.PointCloud())

# vis.run()
# vis2.run()

colors = np.random.rand(8,3)
def visualize(data, label):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    color = np.zeros((len(data), 3))
    for j in range(8):
        color[label == j] += colors[j]

    pcd.colors = o3d.utility.Vector3dVector(color)

    return pcd

def visualize_res():
    loader = DataLoader(Dales('cuda', 25, 4096), batch_size = 8)
    tiles = np.load(os.path.join("data", "Dales" , f"dales_tt_25_4096.npz"))
    data = tiles['x'].reshape(-1, 3)
    label = tiles['y'].reshape(-1)

    model = PointTransformerSeg()
    model_name = 1
    model.load_state_dict(torch.load(os.path.join("model", "best", f"model_{model_name}.pt")))
    # model.eval()

    _,_,_,preds = test_loop(loader)
    preds = np.asarray(preds).reshape(-1)

    shifted_x_data = data
    pcd = visualize(data, label)

    # data[:, 0] += 550
    # pcd2 = visualize(data, preds)
    # # print(shifted_x_data)
    # print(np.unique(preds), preds.shape)
    # print(np.unique(label, return_counts=True), label.shape)





    vis1 = o3d.visualization.Visualizer()
    vis1.create_window()
    vis1.add_geometry(pcd)
    # vis1.add_geometry(pcd2)
    vis1.run()


tiles = np.load(os.path.join("data", "Dales" , f"dales_tt_25_4096.npz"))
# data = tiles['x'].reshape(-1, 3)
data = tiles['x'][:400].reshape(-1, 3)

# mn = np.min(data, axis = 1, keepdims=True)
# mx = np.max(data, axis = 1, keepdims=True)
# data = (data - mn)/(mx - mn)
# print(data.shape)

pcd = visualize(data, tiles['y'][:400].reshape(-1))

o3d.visualization.draw_geometries([pcd])







