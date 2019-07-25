# %%
import json
import os
import subprocess

import numpy as np
import pandas as pd

# %%


def flatten_vrd_relationship(img, relationship, objects, predicates):
    """Create a per-relationship entry from a per-image entry JSON."""
    new_relationship_dict = {}
    new_relationship_dict["subject_category"] = objects[
        relationship["subject"]["category"]
    ]
    new_relationship_dict["object_category"] = objects[
        relationship["object"]["category"]
    ]
    new_relationship_dict["subject_bbox"] = relationship["subject"]["bbox"]
    new_relationship_dict["object_bbox"] = relationship["object"]["bbox"]

    if predicates[relationship["predicate"]] == "ride":
        new_relationship_dict["label"] = 0
    elif predicates[relationship["predicate"]] == "carry":
        new_relationship_dict["label"] = 1
    else:
        new_relationship_dict["label"] = 2

    new_relationship_dict["source_img"] = img

    return new_relationship_dict


# %%
def vrd_to_pandas(
    relationships_set, objects, predicates, list_of_predicates, keys_list=None
):
    """Create Pandas DataFrame from JSON of relationships."""
    relationships = []

    for img in relationships_set:
        if (keys_list is None) or (img in keys_list):
            img_relationships = relationships_set[img]
            for relationship in img_relationships:
                predicate_idx = relationship["predicate"]
                if predicates[predicate_idx] in list_of_predicates:
                    relationships.append(
                        flatten_vrd_relationship(img, relationship, objects, predicates)
                    )
        else:
            continue
    return pd.DataFrame.from_dict(relationships)


# %%
def load_vrd_data():
    """Download and load Pandas DataFrame of VRD relationships.

    NOTE: Only loads semantic relationship examples.
    """
    subprocess.call("bash scene_graph/download_data.sh", shell=True)

    relationships_train = json.load(open("scene_graph/data/VRD/annotations_train.json"))
    relationships_test = json.load(open("scene_graph/data/VRD/annotations_test.json"))

    objects = json.load(open("scene_graph/data/VRD/objects.json"))
    predicates = json.load(open("scene_graph/data/VRD/predicates.json"))
    semantic_predicates = [
        "carry",
        "cover",
        "fly",
        "look",
        "lying on",
        "park on",
        "sit on",
        "stand on",
        "ride",
    ]

    np.random.seed(123)
    val_idx = list(np.random.choice(len(relationships_train), 1000, replace=False))
    relationships_val = {
        key: value
        for i, (key, value) in enumerate(relationships_train.items())
        if i in val_idx
    }
    relationships_train = {
        key: value
        for i, (key, value) in enumerate(relationships_train.items())
        if i not in val_idx
    }

    # TODO: hack to work with small sample of data for tox
    if os.path.isdir("scene_graph/data/VRD/sg_dataset/samples"):
        # pass in list of images as keys_list
        keys_list = os.listdir("scene_graph/data/VRD/sg_dataset/samples")
        test_df = vrd_to_pandas(
            relationships_test,
            objects,
            predicates,
            list_of_predicates=semantic_predicates,
            keys_list=keys_list,
        )
        return test_df, test_df, test_df
    elif os.path.isdir("scene_graph/data/VRD/sg_dataset/sg_train_images"):
        train_df = vrd_to_pandas(
            relationships_train,
            objects,
            predicates,
            list_of_predicates=semantic_predicates,
        )
        train_df["labels"] = -1 * np.ones(len(train_df))
        valid_df = vrd_to_pandas(
            relationships_val,
            objects,
            predicates,
            list_of_predicates=semantic_predicates,
        )
        test_df = vrd_to_pandas(
            relationships_test,
            objects,
            predicates,
            list_of_predicates=semantic_predicates,
        )
        return train_df, valid_df, test_df
