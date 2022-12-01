import os
import xmltodict
import json

num_null = 0
null_list = []
base_dir = "./yolodata/labels"

for xml_file in os.listdir(base_dir):
    if "xml" not in xml_file:
        continue
    with open(f"{base_dir}/{xml_file}", encoding="UTF-8") as fd:
        doc = xmltodict.parse(fd.read())
        dic = json.loads(json.dumps(doc))


    for label in dic["annotations"]["image"]:
        fname = label["@name"]
        print(fname)
        id = label["@id"]
        width, height = label["@width"], label["@height"]

        bbox = label["box"]

        if not os.path.isfile(f"./yolodata/images/val/{fname}"):
            num_null += 1
            null_list.append(fname)
            # print(fname)
