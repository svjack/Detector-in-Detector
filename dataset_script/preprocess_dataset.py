import tensorflow as tf
tf.enable_eager_execution()

from luminoth.utils.bbox_overlap import bbox_overlap_tf

import json
import pandas as pd
from collections import defaultdict

id_category_dict = {
    1: "person",
    2: "face",
    3: "hand",
}

def single_tf_iou_filter(gt_boxes, main_part_label = 1):
    main_part_gt_boxes = tf.boolean_mask(gt_boxes ,tf.equal(gt_boxes[...,1], main_part_label))
    not_main_part_gt_boxes = tf.boolean_mask(gt_boxes ,tf.logical_not(tf.equal(gt_boxes[...,1], main_part_label)))

    #### [num-main, num-part]
    iou_tensor = bbox_overlap_tf(main_part_gt_boxes[:, -4:], not_main_part_gt_boxes[:, -4:])
    total_intersection_num = tf.reduce_sum(tf.reshape(tf.sign(tf.reduce_sum(tf.cast(tf.greater(iou_tensor, 0.0), tf.float32), axis=1)), [-1]),axis=0)

    return total_intersection_num


def json_to_csv(json_path, filter_overlap_num = 0):

    with open(json_path, "r", encoding="utf-8") as f:
        anno_json = json.load(f)
    idx_hash_mapping = dict(map(lambda inner_dict: (inner_dict["id"], inner_dict["file_name"]),anno_json["images"]))

    "ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside"
    anno_list_of_dict = anno_json["annotations"]
    req_dict = defaultdict(list)
    for anno_dict in anno_list_of_dict:
        if anno_dict["category_id"] not in id_category_dict:
            continue

        ImageID = anno_dict["image_id"]
        x, y, w, h = anno_dict["bbox"]
        XMin,YMin,XMax,YMax = x, y, x + w - 1, y + h - 1

        req_dict["ImageID"].append(ImageID)
        req_dict["LabelName"].append(anno_dict["category_id"])

        req_dict["XMin"].append(XMin)
        req_dict["YMin"].append(YMin)
        req_dict["XMax"].append(XMax)
        req_dict["YMax"].append(YMax)

    req_df = pd.DataFrame.from_dict(req_dict)

    all_overlap_num_list = []
    overlap_cnt_dict = defaultdict(list)
    for r_idx ,(idx, group_df) in enumerate(req_df.groupby("ImageID")):
        group_data = group_df[["ImageID", "LabelName", "XMin", "YMin", "XMax", "YMax"]].values
        overlap_num = single_tf_iou_filter(group_data)

        image_id = group_data[0, 0]
        all_overlap_num_list.append(overlap_num.numpy())
        overlap_cnt_dict["ImageID"].append(image_id)
        overlap_cnt_dict["overlap_num"].append(overlap_num.numpy())

        if r_idx % 10 == 0:
            print("r_idx : {} {}".format(r_idx, overlap_num))

    overlap_cnt_df = pd.DataFrame.from_dict(overlap_cnt_dict)
    req_df = overlap_cnt_df.merge(req_df, on="ImageID", how = "inner")
    req_df = req_df[req_df["overlap_num"] > filter_overlap_num]
    req_df["ImageID"] = req_df["ImageID"].apply(lambda imageid: idx_hash_mapping[imageid])
    req_df["LabelName"] = req_df["LabelName"].apply(lambda labelname: id_category_dict[labelname])

    #### image_id,xmin,ymin,xmax,ymax,label
    req_df["image_id"] = req_df["ImageID"]
    req_df["xmin"] = req_df["XMin"].astype(int)
    req_df["ymin"] = req_df["YMin"].astype(int)
    req_df["xmax"] = req_df["XMax"].astype(int)
    req_df["ymax"] = req_df["YMax"].astype(int)
    req_df["label"] = req_df["LabelName"]
    req_df[["image_id","xmin","ymin","xmax","ymax","label"]].to_csv("filtered_train.csv", encoding="utf-8",
                                                                    index=False)

if __name__ == "__main__":
    json_to_csv(
        json_path = r"C:\Coding\Python\Priv_personpart\Json_Annos\privpersonpart_train.json",
    )