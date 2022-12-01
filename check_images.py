import os
import pickle
from typing import Dict
from PIL import Image
from matplotlib import pyplot as plt

# Load data (deserialize)
if os.path.isfile("hist_dicts.pickle"):
    with open('hist_dicts.pickle', 'rb') as handle:
        unserialized_data = pickle.load(handle)

    width_dist_dict = unserialized_data["width_dist_dict"]
    height_dist_dict = unserialized_data["height_dist_dict"]
    ratio_dist_dict = unserialized_data["ratio_dist_dict"]

else:
    base_dir = "yolodata/images/val"
    img_list = os.listdir(base_dir)
    height_dist_dict: Dict[int, int] = {}
    width_dist_dict: Dict[int, int] = {}
    ratio_dist_dict: Dict[float, int] = {}

    for img in img_list:
        im = Image.open(f"{base_dir}/{img}")
        print(img)

        if im.height in height_dist_dict.keys():
            height_dist_dict[im.height] += 1
        else:
            height_dist_dict[im.height] = 1

        if im.width in width_dist_dict.keys():
            width_dist_dict[im.width] += 1
        else:
            width_dist_dict[im.width] = 1

        if im.height > im.width:
            ratio = im.height / im.width
        else:
            ratio = im.width / im.height

        if ratio in ratio_dist_dict.keys():
            ratio_dist_dict[ratio] += 1
        else:
            ratio_dist_dict[ratio] = 1

    hist_dicts: Dict[str, Dict] = {
        "width_dist_dict": width_dist_dict,
        "height_dist_dict": height_dist_dict,
        "ratio_dist_dict": ratio_dist_dict
    }

    # Store data (serialize)
    with open('hist_dicts.pickle', 'wb') as handle:
        pickle.dump(hist_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)

fig = plt.figure(figsize=(15, 30))

plt.subplot(311)
plt.bar(list(width_dist_dict.keys()), list(width_dist_dict.values()))
plt.xlabel("image width")
plt.ylabel("# of samples")
plt.title("Width distribution")

plt.subplot(312)
plt.bar(list(height_dist_dict.keys()), list(height_dist_dict.values()))
plt.xlabel("image height")
plt.ylabel("# of samples")
plt.title("Height distribution")

plt.subplot(313)
plt.bar(list(ratio_dist_dict.keys()), list(ratio_dist_dict.values()))
plt.xlabel("image width-height ratio")
plt.ylabel("# of samples")
plt.title("Ratio distribution")

plt.show()
