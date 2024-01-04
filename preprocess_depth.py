import os
import numpy as np
import sys
import re

def read_pfm(path):
    """Read pfm file.

    Args:
        path (str): path to file

    Returns:
        tuple: (data, scale)
    """
    with open(path, "rb") as file:

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header.decode("ascii") == "PF":
            color = True
        elif header.decode("ascii") == "Pf":
            color = False
        else:
            raise Exception("Not a PFM file: " + path)

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception("Malformed PFM header.")

        scale = float(file.readline().decode("ascii").rstrip())
        if scale < 0:
            # little-endian
            endian = "<"
            scale = -scale
        else:
            # big-endian
            endian = ">"

        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)

        return data


def build_depth_map(basedir, obj):
    file_path = os.path.join(basedir, obj)
    files = os.listdir(file_path)
    files.sort()
    depth_map = []
    for file in files:
        if file.endswith('pfm'):
            data = read_pfm(os.path.join(file_path, file))
            depth_map.append(data)
        
    depth_map = np.stack(depth_map)

    np.save(f"./preprocess/depth_map/defocus{obj}.npy", depth_map)


basedir = sys.argv[1]
obj = sys.argv[2]
os.makedirs("./preprocess/depth_map", exist_ok=True)

if obj == "all":
    for obj in ["cake", "caps", "cisco", "coral", "cups", "cupcake", "daisy", "sausage", "seal", "tools", "cozy2room", "factory", "pool", "tanabata", "wine"]:
        print(obj)
        build_depth_map(basedir, obj)
else:
    build_depth_map(basedir, obj)