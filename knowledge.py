import numpy as np
import time
import os
import sys
import csv
import json
import math

# select paths of the lookup table and cropped knowledge graph
path_project = os.path.abspath(os.path.join(__file__, "../.."))
lookup_filepath = os.path.join(path_project, "Semantic Consistency/KG_lookup_55.csv")
cropped_filepath = os.path.join(path_project, "Semantic Consistency/KG_crop_55.csv")
save_filepath = os.path.join(path_project, "Semantic Consistency/Stored matrices/CM_kg_55_info.json")

class NumpyEncoder(json.JSONEncoder):
    """
    Convert Numpy arrays to JSON
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def convert_91_to_80(S):
    """
    Set the 11 irrelevant COCO classes to 0 in the matrix
    """
    leftouts = [11, 25, 28, 29, 44, 65, 67, 68, 70, 82, 90]
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            if i in leftouts or j in leftouts:
                S[i, j] = 0
    return S

np.set_printoptions(threshold=sys.maxsize)  # don't truncate printing
start = time.time()

input_graph = []

# Load in the relevant assertions from the csv file
with open(cropped_filepath, newline='', encoding="utf8") as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
    for row in spamreader:
        c1idx, c2idx, w = int(row[4]), int(row[5]), float(row[3])
        input_graph.append((c1idx, c2idx, w))

# Create a concept table mapping for integer conversion
concept_table = {}
with open(lookup_filepath, newline='', encoding="utf8") as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
    for row in spamreader:
        concept, number = row[0], int(row[1])
        concept_table[concept] = number

# Define COCO and VOC concepts
interesting_concepts = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                        'traffic_light', 'fire_hydrant', 'street_sign', 'stop_sign', 'parking_meter', 'bench',
                        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                        'hat', 'backpack', 'umbrella', 'shoe', 'eyeglasses', 'handbag', 'tie', 'suitcase',
                        'frisbee', 'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat', 'baseball_glove',
                        'skateboard', 'surfboard', 'tennis_racket', 'bottle', 'plate', 'wine_glass', 'cup',
                        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
                        'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted_plant', 'bed',
                        'mirror', 'dining_table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse',
                        'remote', 'keyboard', 'cell_phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
                        'blender', 'book', 'clock', 'vase', 'scissors', 'teddy_bear', 'hair_dryer', 'toothbrush',
                        'hair_brush')

# Generate index list for relevant concepts
list_of_relevant_indexes = [concept_table[concept] for concept in interesting_concepts]

# Create a blank semantic consistency matrix
s = np.zeros((len(list_of_relevant_indexes), len(list_of_relevant_indexes)))

# Populate the matrix using direct relationships from the input graph
for (c1, c2, weight) in input_graph:
    if c1 in list_of_relevant_indexes and c2 in list_of_relevant_indexes:
        idx1, idx2 = list_of_relevant_indexes.index(c1), list_of_relevant_indexes.index(c2)
        s[idx1, idx2] = weight
        s[idx2, idx1] = weight  # Ensure symmetry for undirected graph

# Compute a symmetrical matrix based on (l1,l2) = (l2,l1) = sqrt(l1*l2)
s_sym = np.zeros_like(s)
for i in range(len(list_of_relevant_indexes)):
    for j in range(len(list_of_relevant_indexes)):
        s_sym[i, j] = math.sqrt(s[i, j] * s[j, i])

print("S_sym: ", s_sym)

s_sym = convert_91_to_80(s_sym)  # Use this if filtering for 80 relevant COCO classes

# Store the matrix in a dictionary for JSON storage
KG_COCO_info = {'S': s_sym}

# Extract VOC concepts out of the COCO matrix
s_sym_voc = np.zeros((20, 20))
coco_to_voc = [5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]
for l1 in range(20):
    for l2 in range(20):
        s_sym_voc[l1, l2] = s_sym[coco_to_voc[l1] - 1, coco_to_voc[l2] - 1]

KG_VOC_info = {'S': s_sym_voc}

# Store all information in a dictionary and save to JSON
info = {'KG_COCO_info': KG_COCO_info, 'KG_VOC_info': KG_VOC_info}
with open(save_filepath, 'w') as j:
    json.dump(info, j, cls=NumpyEncoder)

end = time.time()
hours, rem = divmod(end - start, 3600)
minutes, seconds = divmod(rem, 60)
print("Time to determine S without random walk: ")
print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
