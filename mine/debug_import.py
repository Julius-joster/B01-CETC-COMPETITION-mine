import traj_features
print("实际加载的文件:", traj_features.__file__)
print("模块内所有名字:", [n for n in dir(traj_features) if not n.startswith("_")])