import numpy as np
import sonnet as snt
import tensorflow as tf

from luminoth.models.fasterrcnn.rcnn_overload import RCNN
from luminoth.models.fasterrcnn.part_detector import FasterRCNN as PartFasterRCNN

from luminoth.models.fasterrcnn.rpn import RPN
from luminoth.models.base import TruncatedBaseNetwork
from luminoth.utils.anchors import generate_anchors_reference
from luminoth.utils.vars import VAR_LOG_LEVELS, variable_summaries

from copy import deepcopy
from functools import reduce
from luminoth.utils.image import *
from luminoth.utils.bbox_overlap import bbox_overlap_tf
from luminoth.utils.bbox_transform_tf import change_order


#### input of tf.image.crop_and_resize the bbox order is y1 x1 y2 x2

def patch_image(image, bboxes=None, offset_height=0, offset_width=0,
                target_height=None, target_width=None):
    """Gets a patch using tf.image.crop_to_bounding_box and adjusts bboxes

    If patching would leave us with zero bboxes, we return the image and bboxes
    unchanged.

    Args:
        image: Float32 Tensor with shape (H, W, 3).
        bboxes: Tensor with the ground-truth boxes. Shaped (total_boxes, 5).
            The last element in each box is the category label.
        offset_height: Height of the upper-left corner of the patch with
            respect to the original image. Non-negative.
        offset_width: Width of the upper-left corner of the patch with respect
            to the original image. Non-negative.
        target_height: Height of the patch. If set to none, it will be the
            maximum (tf.shape(image)[0] - offset_height - 1). Positive.
        target_width: Width of the patch. If set to none, it will be the
            maximum (tf.shape(image)[1] - offset_width - 1). Positive.

    Returns:
        image: Patch of the original image.
        bboxes: Adjusted bboxes (only those whose centers are inside the
            patch). The key isn't set if bboxes is None.
    """
    # TODO: make this function safe with respect to senseless inputs (i.e
    # having an offset_height that's larger than tf.shape(image)[0], etc.)
    # As of now we only use it inside random_patch, which already makes sure
    # the arguments are legal.
    im_shape = tf.shape(image)
    if target_height is None:
        target_height = (im_shape[0] - offset_height - 1)
    if target_width is None:
        target_width = (im_shape[1] - offset_width - 1)

    new_image = tf.image.crop_to_bounding_box(
        image,
        offset_height=offset_height, offset_width=offset_width,
        target_height=target_height, target_width=target_width
    )
    patch_shape = tf.shape(new_image)

    # Return if we didn't have bboxes.
    if bboxes is None:
        # Resize the patch to the original image's size. This is to make sure
        # we respect restrictions in image size in the models.
        new_image_resized = tf.image.resize_images(
            new_image, im_shape[:2],
            method=tf.image.ResizeMethod.BILINEAR
        )
        return_dict = {'image': new_image_resized}
        return return_dict


    masked_bboxes = bboxes


    new_bboxes_unclipped = tf.concat(
        [
            tf.subtract(masked_bboxes[:, 0:1], offset_width),
            tf.subtract(masked_bboxes[:, 1:2], offset_height),
            tf.subtract(masked_bboxes[:, 2:3], offset_width),
            tf.subtract(masked_bboxes[:, 3:4], offset_height),
        ],
        axis=1,
    )

    # Finally, we clip the boxes and add back the labels.
    new_bboxes = tf.concat(
        [
            tf.to_int32(
                clip_boxes(
                    new_bboxes_unclipped,
                    imshape=patch_shape[:2]
                ),
            ),
            masked_bboxes[:, 4:]
        ],
        axis=1
    )

    return {
        "image": new_image,
        "bboxes": new_bboxes
    }

class DetectorInDetector(snt.AbstractModule):
    """Faster RCNN Network module

    Builds the Faster RCNN network architecture using different submodules.
    Calculates the total loss of the model based on the different losses by
    each of the submodules.

    It is also responsible for building the anchor reference which is used in
    graph for generating the dynamic anchors.
    """
    def edit_main_config_into_PartDetector_config(self, config):
        #### some others config edit will be add in the feature.

        req_config = deepcopy(config)
        req_config.model.network.num_classes = config.model.network.num_classes - 1

        return req_config

    def __init__(self, config, name='fasterrcnn'):
        super(DetectorInDetector, self).__init__(name=name)

        # Main configuration object, it holds not only the necessary
        # information for this module but also configuration for each of the
        # different submodules.
        self._config = config

        #### some settings should add in config
        self._main_part_label = config.model.main_part_label
        self._main_part_prob_threshold = config.model.main_part_prob_threshold

        ####

        # Total number of classes to classify. If not using RCNN then it is not
        # used. TODO: Make it *more* optional.
        self._num_classes = config.model.network.num_classes

        # Generate network with RCNN thus allowing for classification of
        # objects and not just finding them.
        self._with_rcnn = config.model.network.with_rcnn

        # Turn on debug mode with returns more Tensors which can be used for
        # better visualization and (of course) debugging.
        self._debug = config.train.debug
        self._seed = config.train.seed

        # Anchor config, check out the docs of base_config.yml for a better
        # understanding of how anchors work.
        self._anchor_base_size = config.model.anchors.base_size
        self._anchor_scales = np.array(config.model.anchors.scales)
        self._anchor_ratios = np.array(config.model.anchors.ratios)
        self._anchor_stride = config.model.anchors.stride

        # Anchor reference for building dynamic anchors for each image in the
        # computation graph.
        self._anchor_reference = generate_anchors_reference(
            self._anchor_base_size, self._anchor_ratios, self._anchor_scales
        )

        # Total number of anchors per point.
        self._num_anchors = self._anchor_reference.shape[0]

        # Weights used to sum each of the losses of the submodules
        self._rpn_cls_loss_weight = config.model.loss.rpn_cls_loss_weight
        self._rpn_reg_loss_weight = config.model.loss.rpn_reg_loss_weights

        self._rcnn_cls_loss_weight = config.model.loss.rcnn_cls_loss_weight
        self._rcnn_reg_loss_weight = config.model.loss.rcnn_reg_loss_weights
        self._losses_collections = ['fastercnn_losses']

        # We want the pretrained model to be outside the FasterRCNN name scope.
        self.base_network = TruncatedBaseNetwork(config.model.base_network)

        #### init of PartFasterRCNN
        partdetector_config = self.edit_main_config_into_PartDetector_config(config)
        self.partdetetor = PartFasterRCNN(partdetector_config)

        self._class_max_detections = config.model.rcnn.proposals.class_max_detections
        self._class_nms_threshold = config.model.rcnn.proposals.class_nms_threshold
        self._total_max_detections = config.model.rcnn.proposals.total_max_detections

    def iter_unify_layer(self, inputs, is_training = False, layer_num = 3):
        for i in range(layer_num):
            if i != layer_num - 1:
                inputs = self.unify_layer(inputs, is_training, return_final=False)
            else:
                inputs = self.unify_layer(inputs, is_training, return_final=True)

        return inputs

    def unify_layer(self, inputs, is_training = False, return_final = False, filters = 256):
        conv_part = tf.layers.conv2d(
            inputs = inputs,
            filters = 256,
            kernel_size = (3, 3),
            strides=(1, 1),
            padding='same',
        )

        if return_final:
            return tf.nn.relu(tf.layers.batch_normalization(
                inputs=conv_part, trainable=is_training
            ))
        else:
            return conv_part


    def padding_and_slice_PartDetector_features(self, body_pred_feature, body_ground_truth_feature, fixed_slice_size = 7):
        if body_ground_truth_feature is not None:
            concat_t = tf.concat([body_pred_feature, body_ground_truth_feature], axis = 0)
            fixed_slice_size = tf.minimum(tf.shape(concat_t)[0], fixed_slice_size)
            return (fixed_slice_size ,tf.slice(concat_t,
                                               begin=[0, 0], size = [fixed_slice_size, -1]))
        else:
            concat_t = body_pred_feature
            have_num = tf.minimum(tf.shape(concat_t)[0], fixed_slice_size)
            concat_t = tf.cond(tf.greater(fixed_slice_size, have_num),
                               true_fn=lambda : tf.concat([concat_t, tf.zeros(shape=[fixed_slice_size - have_num, tf.shape(concat_t)[1]], dtype=tf.float32)], axis=0),
                               false_fn=lambda : tf.slice(concat_t,
                                                          begin=[0, 0], size = [fixed_slice_size, -1]))
            return (fixed_slice_size, concat_t)

    #### gt_boxes contain the part_annoations and (BodyDetector's top some body detections or ground truth body annotations,
    # the latter for padding require. merge this two conclusions and padding and slice to fixed size for __call___ the
    # PartDetector, the first version is only use ground truth body annotation's)
    def generate_PartDetector_features(self, input_image, input_feature, gt_boxes, only_main_part_boxes = False):
        assert only_main_part_boxes in [True, False]
        main_part_label = self._main_part_label

        image_h, image_w = tf.split(tf.shape(input_image)[0:2], num_or_size_splits=2)
        feature_h, feature_w = tf.split(tf.shape(input_feature)[1:3], num_or_size_splits=2)

        main_part_gt_boxes = tf.boolean_mask(gt_boxes ,tf.reshape(tf.equal(gt_boxes[...,-1], main_part_label), [-1]))
        if not only_main_part_boxes:
            not_main_part_gt_boxes = tf.boolean_mask(gt_boxes ,tf.reshape(tf.logical_not(tf.equal(gt_boxes[...,-1], main_part_label)), [-1]))

            iou_tensor = bbox_overlap_tf(main_part_gt_boxes[:, :4], not_main_part_gt_boxes[:, :4])

            reproduce_iou = iou_tensor > tf.constant(0.0, dtype=tf.float32)
            intersection_indexes = tf.where(reproduce_iou)
            intersection_indexes = tf.cast(intersection_indexes, dtype=tf.int32)

        #### total_shape [ 1 + 24 * 24 * 1024 + 7 * 5] = [589860]
        def single_patch_image(patch_dict ,image_resize = (24, 24) ,bboxes_padding_range = 7.0):
            image = patch_dict["image"]
            im_shape = tf.shape(image)
            shape_prod = im_shape[0] * im_shape[1] * im_shape[2]

            image = tf.cond(
                tf.greater(shape_prod, 0),
                true_fn=lambda : tf.image.resize_images(tf.expand_dims(image, 0), size=image_resize),
                false_fn=lambda : tf.zeros(shape=[1 ,24, 24, 256], dtype=tf.float32)
            )

            image = tf.layers.max_pooling2d(
                inputs = image,
                pool_size = (2, 2), strides = (1, 1),
                padding='same',
            )
            image_flatten = tf.reshape(image, [-1])

            if not only_main_part_boxes:
                bboxes = tf.cast(patch_dict["bboxes"], dtype=tf.float32)
                bboxes = bboxes[:tf.cast(bboxes_padding_range, tf.int32), ...]

                num_bboxes = tf.cast(tf.shape(bboxes)[0], tf.float32)
                bboxes_padding = tf.concat([bboxes, tf.zeros(shape=[tf.cast(bboxes_padding_range - num_bboxes, dtype=tf.int32), 5])], axis=0)
                bboxes_flatten = tf.reshape(bboxes_padding, [-1])
                num_bboxes = tf.reshape(num_bboxes, [-1])

                return  tf.concat([num_bboxes, image_flatten, bboxes_flatten], axis=0)

            return image_flatten

        def single_map(main_index):
            t4 = tf.reshape(tf.cast(main_part_gt_boxes[main_index][:4], tf.int32), [-1])
            #return t4

            if not only_main_part_boxes:
                bbox = tf.cast(tf.gather(not_main_part_gt_boxes, tf.reshape(tf.gather(intersection_indexes[:, -1] ,tf.where(tf.equal(intersection_indexes[:, 0], main_index))), [-1])),
                               dtype=tf.int32)
                bbox = bbox[:, :5]
                patch_bbox_conclusion = patch_image(
                    image=input_image, bboxes=bbox,
                    offset_width=t4[0], offset_height=t4[1],
                    target_width=t4[2] - t4[0] + 1, target_height=t4[3] - t4[1] + 1
                )
                bboxes_patched = patch_bbox_conclusion["bboxes"]
            else:
                bboxes_patched = None

            patch_feature_conclusion = tf.slice(input_feature,
                                                begin=[0,
                                                       tf.reshape(tf.cast(tf.cast(t4[1], tf.float32) / tf.cast(image_h, tf.float32) * tf.cast(feature_h, tf.float32), tf.int32), []),
                                                       tf.reshape(tf.cast(tf.cast(t4[0], tf.float32) / tf.cast(image_w, tf.float32) * tf.cast(feature_w, tf.float32), tf.int32), []),
                                                       0],
                                                size=[-1,
                                                      tf.reshape(tf.cast(tf.cast(t4[3] - t4[1], tf.float32)/ tf.cast(image_h, tf.float32) * tf.cast(feature_h, tf.float32), tf.int32), []) ,
                                                      tf.reshape(tf.cast(tf.cast(t4[2] - t4[0], tf.float32) / tf.cast(image_w, tf.float32) * tf.cast(feature_w, tf.float32), tf.int32), []) ,
                                                      -1]
                                                )

            feature_patched = tf.squeeze(patch_feature_conclusion, 0)

            patch_conclusion = {
                "image": feature_patched,
                "bboxes": bboxes_patched
            }

            #### when only_main_part_boxes, single_tensor is only flatten_image, the return final is 5 + flatten_image_dim i.e. 5 + 24 * 24 * 256 = 147461
            single_tensor = single_patch_image(patch_conclusion)

            #### 5 +
            concat_tensor = tf.concat([tf.reshape(main_part_gt_boxes[main_index], [-1]) ,single_tensor], axis=0)
            return concat_tensor

        # ?????
        return tf.cond(
            tf.greater(tf.reduce_sum(tf.reshape(tf.cast(reproduce_iou, tf.float32), [-1])), 0.0),
            true_fn=lambda :tf.map_fn(
                single_map, intersection_indexes[:, 0], dtype=tf.float32
            ),
            false_fn=lambda :tf.zeros([0, 147497], dtype=tf.float32)
        ) if not only_main_part_boxes else tf.map_fn(single_map, tf.cast(tf.range(tf.shape(main_part_gt_boxes)[0]), tf.int32),
                                                     dtype=tf.float32)

    def inverse_transform_labels(self ,labels):
        ##### reverse to retrieve into require.
        labels = labels + tf.nn.relu(tf.sign(tf.sign(self._main_part_label - labels) * -1 + 1))
        return labels

    #### input_feature is a 1d tensor,
    #### encode the label
    def decode_single_unstacked_feature(self, input_feature, only_main_part_boxes = False):
        main_part_ori_bbox = tf.slice(input_feature, begin = [0], size = [5])

        # [ 1 + 24 * 24 * 3 + 3 * 5]
        #  [ 1 + 24 * 24 * 1024 + 7 * 5] = [589860]

        if not only_main_part_boxes:
            num_of_bboxes = tf.cast(tf.squeeze(tf.slice(input_feature, begin = [5], size=[1])), dtype=tf.int32)

            # [24 * 24 *1024 + 7 * 5]
            res_feature = tf.slice(input_feature, begin=[5 + 1], size=[-1])
            image_feature = tf.slice(res_feature, begin=[0], size = [24 * 24 * 256])
            image = tf.reshape(image_feature, [24, 24, 256])
            all_num_of_bboxes = 7
            bboxes = tf.reshape(tf.slice(res_feature, begin = [24 * 24 * 256], size = [-1]), [all_num_of_bboxes, 5])
            gt_boxes_filtered = bboxes[:num_of_bboxes, ...]

            #### encode the label
            transformed_labels = gt_boxes_filtered[...,-1:] - tf.nn.relu(tf.sign(gt_boxes_filtered[...,-1:] - self._main_part_label))

            gt_boxes_filtered_req = tf.concat(
                [gt_boxes_filtered[...,:-1], transformed_labels], -1
            )

            return (main_part_ori_bbox ,image, gt_boxes_filtered_req)
            #return (main_part_ori_bbox ,image, gt_boxes_filtered)
        else:
            res_feature = tf.slice(input_feature, begin=[5], size=[-1])
            image_feature = tf.slice(res_feature, begin=[0], size = [24 * 24 * 256])
            image = tf.reshape(image_feature, [24, 24, 256])
            return (main_part_ori_bbox, image)

    #### reduce and decode the label
    def reduce_prediction_dict_list(self, all_dict_list):
        def map_obj_label_prob_to_main_part(pred_dict, add_outer_bbox = False):
            main_info_dict = pred_dict["main_info"]
            image = main_info_dict["image"]
            main_part_ori_bbox = main_info_dict["main_part_ori_bbox"]

            objects = pred_dict['classification_prediction']['objects']
            objects_labels = pred_dict['classification_prediction']['labels']
            objects_labels_prob = pred_dict['classification_prediction']['probs']

            x1, y1, x2, y2 ,_ = tf.split(main_part_ori_bbox, 5)
            objects = tf.concat([
                objects[:, 0:1] + x1, objects[:, 1:2] + y1, objects[:, 2:3] + x1, objects[:, 3:4] + y1
            ], axis=-1)

            if add_outer_bbox:
                objects = tf.concat([objects, tf.convert_to_tensor([tf.concat([x1, y1, x2, y2], axis=0)], dtype=objects.dtype)], axis=0)
                objects_labels = tf.concat([self.inverse_transform_labels(objects_labels), tf.convert_to_tensor([self._main_part_label], dtype=objects_labels.dtype)], axis=0)
                objects_labels_prob = tf.concat([objects_labels_prob, tf.convert_to_tensor([1.0], dtype=objects_labels_prob.dtype)], axis=0)
                return (objects, objects_labels, objects_labels_prob)

            return (objects, self.inverse_transform_labels(objects_labels), objects_labels_prob)

        def retrieve_main(pred_dict):
            return (pred_dict['classification_prediction']['objects'], pred_dict['classification_prediction']['labels'],
                    pred_dict['classification_prediction']['probs'])

        def reduce_all_before_nms():
            t3_list = []
            for idx, pred_dict in enumerate(all_dict_list):
                if idx == 0:
                    t3 = retrieve_main(pred_dict)
                    #### filter main for tiny eval.
                    #continue
                else:
                    t3 = map_obj_label_prob_to_main_part(pred_dict)
                t3_list.append(t3)

            t3 = tuple(map(lambda idx: tf.concat(list(map(lambda t3: t3[idx], t3_list)), axis=0), range(3)))
            return t3

        def build_without_filter(class_objects, cls_prob, cls_label):
            selected_boxes = []
            selected_probs = []
            selected_labels = []

            # For each class, take the proposals with the class-specific
            # predictions (class scores and bbox regression) and filter accordingly
            # (valid area, min probability score and NMS).
            for class_id in range(self._num_classes):
                # Apply the class-specific transformations to the proposals to
                # obtain the current class' prediction.
                label_filer = tf.reshape(tf.where(tf.equal(class_id, cls_label)), [-1])

                class_objects_filtered, cls_prob_filtered = map(lambda x: tf.gather(x, label_filer), [class_objects, cls_prob])

                # Filter objects based on the min probability threshold and on them
                # having a valid area.

                #### for filter trivial padding conclusion
                prob_filter = tf.greater_equal(
                    cls_prob_filtered, 0.2
                )

                (x_min, y_min, x_max, y_max) = tf.unstack(class_objects_filtered, axis=1)

                area_filter = tf.greater(
                    tf.maximum(x_max - x_min, 0.0)
                    * tf.maximum(y_max - y_min, 0.0),
                    0.0
                )

                object_filter = tf.logical_and(area_filter, prob_filter)

                class_objects_filtered = tf.boolean_mask(class_objects_filtered, object_filter)
                cls_prob_filtered = tf.boolean_mask(cls_prob_filtered, object_filter)

                # We have to use the TensorFlow's bounding box convention to use
                # the included function for NMS.
                class_objects_tf = change_order(class_objects_filtered)

                # Apply class NMS.
                class_selected_idx = tf.image.non_max_suppression(
                    class_objects_tf, cls_prob_filtered, self._class_max_detections,
                    iou_threshold=self._class_nms_threshold
                )

                # Using NMS resulting indices, gather values from Tensors.
                class_objects_tf = tf.gather(class_objects_tf, class_selected_idx)
                class_prob = tf.gather(cls_prob_filtered, class_selected_idx)

                # Revert to our bbox convention.
                class_objects_tf = change_order(class_objects_tf)

                # We append values to a regular list which will later be
                # transformed to a proper Tensor.
                selected_boxes.append(class_objects_tf)
                selected_probs.append(class_prob)
                # In the case of the class_id, since it is a loop on classes, we
                # already have a fixed class_id. We use `tf.tile` to create that
                # Tensor with the total number of indices returned by the NMS.

                selected_labels.append(
                    tf.tile([class_id], [tf.shape(class_selected_idx)[0]])
                )

            # We use concat (axis=0) to generate a Tensor where the rows are
            # stacked on top of each other
            objects = tf.concat(selected_boxes, axis=0)
            proposal_label = tf.concat(selected_labels, axis=0)
            proposal_label_prob = tf.concat(selected_probs, axis=0)

            # Get top-k detections of all classes.
            k = tf.minimum(
                self._total_max_detections,
                tf.shape(proposal_label_prob)[0]
            )
            top_k = tf.nn.top_k(proposal_label_prob, k=k)
            top_k_proposal_label_prob = top_k.values
            top_k_objects = tf.gather(objects, top_k.indices)
            top_k_proposal_label = tf.gather(proposal_label, top_k.indices)

            return (top_k_objects, top_k_proposal_label, top_k_proposal_label_prob)

        def apply_nms_to_t3(t3 = reduce_all_before_nms()):
            obj, label, prob = t3
            t3 = build_without_filter(class_objects=obj, cls_label=label, cls_prob=prob)
            return t3

        return apply_nms_to_t3()

    def add_main_to_reduce(self, all_dict_list):
        main_pred_dict = all_dict_list[0]
        obj, label, prob = self.reduce_prediction_dict_list(all_dict_list)
        main_pred_dict['classification_prediction']['objects'] = obj
        main_pred_dict['classification_prediction']['labels'] = label
        main_pred_dict['classification_prediction']['probs'] = prob
        return main_pred_dict

    def _build(self, image, gt_boxes=None, is_training=False):
        """
        Returns bounding boxes and classification probabilities.

        Args:
            image: A tensor with the image.
                Its shape should be `(height, width, 3)`.
            gt_boxes: A tensor with all the ground truth boxes of that image.
                Its shape should be `(num_gt_boxes, 5)`
                Where for each gt box we have (x1, y1, x2, y2, label),
                in that order.
            is_training: A boolean to whether or not it is used for training.

        Returns:
            classification_prob: A tensor with the softmax probability for
                each of the bounding boxes found in the image.
                Its shape should be: (num_bboxes, num_categories + 1)
            classification_bbox: A tensor with the bounding boxes found.
                It's shape should be: (num_bboxes, 4). For each of the bboxes
                we have (x1, y1, x2, y2)
        """

        #### use variable_scope to split BodyDetector and PartDetector



        if gt_boxes is not None:
            gt_boxes = tf.cast(gt_boxes, tf.float32)
        # A Tensor with the feature map for the image,
        # its shape should be `(feature_height, feature_width, 512)`.
        # The shape depends of the pretrained network in use.

        # Set rank and last dimension before using base network
        # TODO: Why does it loose information when using queue?
        image.set_shape((None, None, 3))

        conv_feature_map = self.base_network(
            tf.expand_dims(image, 0), is_training=is_training
        )

        C4 = conv_feature_map
        with tf.variable_scope("C5"):
            C5 = self.iter_unify_layer(C4, is_training=is_training)
            #C5 = self.unify_layer(C4, is_training=is_training)

        with tf.variable_scope("Head_body_part"):
            Head_body_part = self.iter_unify_layer(C5, is_training=is_training)
            #Head_body_part = self.unify_layer(C5, is_training=is_training)

        with tf.variable_scope("Head_hf_part"):
            Head_hf_part = self.iter_unify_layer(C5, is_training=is_training)
            #Head_hf_part = self.unify_layer(C5, is_training=is_training)

        with tf.variable_scope("Head_hf_part_conv"):
            Head_hf_part_conv = self.iter_unify_layer(
                Head_hf_part, is_training=is_training
            )

        # The RPN submodule which generates proposals of objects.
        self._rpn = RPN(
            self._num_anchors, self._config.model.rpn,
            debug=self._debug, seed=self._seed
        )

        if self._with_rcnn:
            # The RCNN submodule which classifies RPN's proposals and
            # classifies them as background or a specific class.
            self._rcnn = RCNN(
                self._num_classes, self._config.model.rcnn,
                debug=self._debug, seed=self._seed,
                name="__rcnn__1"
            )

        image_shape = tf.shape(image)[0:2]

        variable_summaries(
            conv_feature_map, 'conv_feature_map', 'reduced'
        )

        # Generate anchors for the image based on the anchor reference.
        all_anchors_1 = self._generate_anchors(tf.shape(conv_feature_map))

        rpn_1_prediction = self._rpn(
            conv_feature_map, image_shape, all_anchors_1,
            gt_boxes=gt_boxes, is_training=is_training
        )

        prediction_1_dict = {
            'rpn_prediction': rpn_1_prediction,
        }

        if self._debug:
            prediction_1_dict['image'] = image
            prediction_1_dict['image_shape'] = image_shape
            prediction_1_dict['all_anchors'] = all_anchors_1
            prediction_1_dict['anchor_reference'] = tf.convert_to_tensor(
                self._anchor_reference
            )
            if gt_boxes is not None:
                prediction_1_dict['gt_boxes'] = gt_boxes
            prediction_1_dict['conv_feature_map'] = conv_feature_map

        if self._with_rcnn:
            proposals = tf.stop_gradient(rpn_1_prediction['proposals'])

            rpn_1_proposals = proposals

            classification_pred = self._rcnn(
                Head_body_part, proposals,
                image_shape, self.base_network,
                gt_boxes=gt_boxes, is_training=is_training
            )

            #### retrieve req from classification_pred
            without_filter_dict = classification_pred["without_filter_dict"]

            objects_1_all = without_filter_dict["objects"]
            labels_1_all = without_filter_dict["proposal_label"]
            probs_1_all = without_filter_dict["proposal_label_prob"]

            objects_1 = classification_pred["objects"]
            labels_1 = classification_pred["labels"]
            probs_1 = classification_pred["probs"]

            prediction_1_dict['objects'] = objects_1
            prediction_1_dict['labels'] = labels_1
            prediction_1_dict['probs'] = probs_1

            top_indices = tf.nn.top_k(tf.cast(1 - tf.sign(tf.abs(labels_1_all - self._main_part_label)), dtype=tf.float32) + probs_1_all,
                                      k = tf.shape(labels_1_all)[0]).indices

            objects_1_sorted = tf.gather(objects_1_all ,top_indices)
            filter_num = tf.minimum(tf.shape(objects_1_sorted)[0], 7)

            objects_1_filtered = tf.slice(objects_1_sorted, begin=[0, 0], size=[filter_num, 4])
            #### expand with label [?, 4] -> [?, 5]
            objects_1_filtered = tf.concat([objects_1_filtered, tf.fill([tf.shape(objects_1_filtered)[0], 1], value=tf.convert_to_tensor(self._main_part_label,
                                                                                                                                         dtype=tf.float32))],
                                           axis=-1)

            prediction_1_dict['classification_prediction'] = classification_pred

            if gt_boxes is not None:
                body_feature_ground_truth = self.generate_PartDetector_features(
                    input_image=image, input_feature=Head_hf_part, gt_boxes = gt_boxes, only_main_part_boxes=False
                )
                body_feature_pred = self.generate_PartDetector_features(
                    input_image=image, input_feature=Head_hf_part, gt_boxes=tf.concat([tf.gather(gt_boxes, tf.reshape(tf.where(tf.not_equal(gt_boxes[:, -1], self._main_part_label)), [-1])),
                                                                                       objects_1_filtered], axis=0)
                    ,only_main_part_boxes=False)
            else:
                body_feature_ground_truth = None
                body_feature_pred = self.generate_PartDetector_features(
                    input_image=image, input_feature=Head_hf_part, gt_boxes=objects_1_filtered,
                    only_main_part_boxes=True
                )

            #### use as fake placeholder
            if gt_boxes is not None:
                body_feature_pred = tf.reshape(body_feature_pred, [-1, tf.shape(body_feature_ground_truth)[-1]])
            else:
                body_feature_pred = tf.reshape(body_feature_pred, [-1, 147461])

            #### unstack it in firxt dim and "map reduce" it on modified faster-rcnn
            #### but the input ground truth label should perform label remapping is the "decoder" of single feature
            fixed_sliced_size ,PartDetector_feature_stacked = self.padding_and_slice_PartDetector_features(body_pred_feature=body_feature_pred, body_ground_truth_feature=body_feature_ground_truth)
            PartDetector_feature_stacked = tf.slice(PartDetector_feature_stacked, begin=[0, 0], size=[fixed_sliced_size, -1])

            if gt_boxes is not None:
                PartDetector_feature_stacked = tf.gather(PartDetector_feature_stacked, tf.random_shuffle(tf.range(fixed_sliced_size)))
                PartDetector_feature_stacked = tf.reshape(PartDetector_feature_stacked, [fixed_sliced_size, -1])
                PartDetector_feature_unstacked = [PartDetector_feature_stacked[0,...]]
            else:
                PartDetector_feature_unstacked = tf.unstack(PartDetector_feature_stacked, axis=0)
            partdetector_dict_list = []

            for single_partdetector_feature in PartDetector_feature_unstacked:
                if gt_boxes is not None:
                    main_part_ori_bbox ,cropped_feature, cropped_bboxes  = self.decode_single_unstacked_feature(input_feature=single_partdetector_feature, only_main_part_boxes = True if gt_boxes is None else False)
                else:
                    main_part_ori_bbox, cropped_feature = self.decode_single_unstacked_feature(input_feature=single_partdetector_feature, only_main_part_boxes = True if gt_boxes is None else False)
                    cropped_bboxes = None

                x1, y1, x2, y2 ,_ = tf.split(main_part_ori_bbox, 5)
                x1, y1, x2, y2 = map(lambda x: tf.cast(tf.reshape(x, []), tf.int32), [x1, y1, x2, y2])

                cropped_image = tf.image.crop_to_bounding_box(image=image, offset_height=y1, offset_width=x1, target_height=y2 - y1 + 1, target_width=x2 - x1 + 1)
                cropped_feature = tf.expand_dims(cropped_feature, 0)

                input_feature = Head_hf_part_conv
                image_h, image_w = tf.split(tf.shape(image)[0:2], num_or_size_splits=2)
                feature_h, feature_w = tf.split(tf.shape(input_feature)[1:3], num_or_size_splits=2)

                t4 = [x1, y1, x2, y2]
                Head_hf_part_conv = tf.slice(input_feature,
                                             begin=[0,
                                                    tf.reshape(tf.cast(tf.cast(t4[1], tf.float32) / tf.cast(image_h, tf.float32) * tf.cast(feature_h, tf.float32), tf.int32), []),
                                                    tf.reshape(tf.cast(tf.cast(t4[0], tf.float32) / tf.cast(image_w, tf.float32) * tf.cast(feature_w, tf.float32), tf.int32), []),
                                                    0],
                                             size=[-1,
                                                   tf.reshape(tf.cast(tf.cast(t4[3] - t4[1], tf.float32)/ tf.cast(image_h, tf.float32) * tf.cast(feature_h, tf.float32), tf.int32), []) ,
                                                   tf.reshape(tf.cast(tf.cast(t4[2] - t4[0], tf.float32) / tf.cast(image_w, tf.float32) * tf.cast(feature_w, tf.float32), tf.int32), []) ,
                                                   256]
                                             )

                #### Head_hf_part_conv  not crop, test the efficiency
                partdetector_dict = self.partdetetor(conv_feature_map = cropped_feature, Head_hf_part_conv = Head_hf_part_conv ,image = cropped_image, gt_boxes = cropped_bboxes, is_training = is_training)

                partdetector_dict["main_info"] = {
                    "image": image,
                    "main_part_ori_bbox": main_part_ori_bbox
                }

                partdetector_dict_list.append(partdetector_dict)

            return [prediction_1_dict] + partdetector_dict_list

    def partial_reduce_pred_list(self, all_dict_list):
        all_dict_list[0] = self.add_main_to_reduce(all_dict_list)
        return all_dict_list

    def loss(self, prediction_dict_list):
        body_prediction = prediction_dict_list[0]
        with tf.variable_scope("body_detector_loss"):
            body_detector_loss = self.single_loss(body_prediction, _rpn=self._rpn, _rcnn=self._rcnn)

        part_prediction_list = prediction_dict_list[1:]
        part_detector_loss_list = []
        for index ,part_prediction in enumerate(part_prediction_list):
            with tf.variable_scope("part_detector_loss_{}".format(index)):
                part_detector_loss = self.single_loss(part_prediction, _rpn=self.partdetetor._rpn, _rcnn=self.partdetetor._rcnn)
                part_detector_loss_list.append(part_detector_loss)

        return body_detector_loss + reduce(lambda a, b: a + b, part_detector_loss_list)

    def single_loss(self, prediction_dict, _rpn, _rcnn, return_all=False):
        """Compute the joint training loss for Faster RCNN.

        Args:
            prediction_dict: The output dictionary of the _build method from
                which we use two different main keys:

                rpn_prediction: A dictionary with the output Tensors from the
                    RPN.
                classification_prediction: A dictionary with the output Tensors
                    from the RCNN.

        Returns:
            If `return_all` is False, a tensor for the total loss. If True, a
            dict with all the internal losses (RPN's, RCNN's, regularization
            and total loss).
        """

        with tf.name_scope('losses'):
            self._rpn = _rpn
            self._rcnn = _rcnn

            rpn_loss_dict = self._rpn.loss(
                prediction_dict['rpn_prediction']
            )

            # Losses have a weight assigned, we multiply by them before saving
            # them.
            rpn_loss_dict['rpn_cls_loss'] = (
                    rpn_loss_dict['rpn_cls_loss'] * self._rpn_cls_loss_weight)
            rpn_loss_dict['rpn_reg_loss'] = (
                    rpn_loss_dict['rpn_reg_loss'] * self._rpn_reg_loss_weight)

            prediction_dict['rpn_loss_dict'] = rpn_loss_dict

            if self._with_rcnn:
                rcnn_loss_dict = self._rcnn.loss(
                    prediction_dict['classification_prediction']
                )

                rcnn_loss_dict['rcnn_cls_loss'] = (
                        rcnn_loss_dict['rcnn_cls_loss'] *
                        self._rcnn_cls_loss_weight
                )
                rcnn_loss_dict['rcnn_reg_loss'] = (
                        rcnn_loss_dict['rcnn_reg_loss'] *
                        self._rcnn_reg_loss_weight
                )

                prediction_dict['rcnn_loss_dict'] = rcnn_loss_dict
            else:
                rcnn_loss_dict = {}

            all_losses_items = (
                    list(rpn_loss_dict.items()) + list(rcnn_loss_dict.items()))

            for loss_name, loss_tensor in all_losses_items:
                tf.summary.scalar(
                    loss_name, loss_tensor,
                    collections=self._losses_collections
                )
                # We add losses to the losses collection instead of manually
                # summing them just in case somebody wants to use it in another
                # place.
                tf.losses.add_loss(loss_tensor)

            # Regularization loss is automatically saved by TensorFlow, we log
            # it differently so we can visualize it independently.
            regularization_loss = tf.losses.get_regularization_loss()
            # Total loss without regularization
            no_reg_loss = tf.losses.get_total_loss(
                add_regularization_losses=False
            )
            total_loss = tf.losses.get_total_loss()

            tf.summary.scalar(
                'total_loss', total_loss,
                collections=self._losses_collections
            )
            tf.summary.scalar(
                'no_reg_loss', no_reg_loss,
                collections=self._losses_collections
            )
            tf.summary.scalar(
                'regularization_loss', regularization_loss,
                collections=self._losses_collections
            )

            if return_all:
                loss_dict = {
                    'total_loss': total_loss,
                    'no_reg_loss': no_reg_loss,
                    'regularization_loss': regularization_loss,
                }

                for loss_name, loss_tensor in all_losses_items:
                    loss_dict[loss_name] = loss_tensor

                return loss_dict

            # We return the total loss, which includes:
            # - rpn loss
            # - rcnn loss (if activated)
            # - regularization loss
            return total_loss

    def _generate_anchors(self, feature_map_shape):
        """Generate anchor for an image.

        Using the feature map, the output of the pretrained network for an
        image, and the anchor_reference generated using the anchor config
        values. We generate a list of anchors.

        Anchors are just fixed bounding boxes of different ratios and sizes
        that are uniformly generated throught the image.

        Args:
            feature_map_shape: Shape of the convolutional feature map used as
                input for the RPN. Should be (batch, height, width, depth).

        Returns:
            all_anchors: A flattened Tensor with all the anchors of shape
                `(num_anchors_per_points * feature_width * feature_height, 4)`
                using the (x1, y1, x2, y2) convention.
        """
        with tf.variable_scope('generate_anchors'):
            grid_width = feature_map_shape[2]  # width
            grid_height = feature_map_shape[1]  # height
            shift_x = tf.range(grid_width) * self._anchor_stride
            shift_y = tf.range(grid_height) * self._anchor_stride
            shift_x, shift_y = tf.meshgrid(shift_x, shift_y)

            shift_x = tf.reshape(shift_x, [-1])
            shift_y = tf.reshape(shift_y, [-1])

            shifts = tf.stack(
                [shift_x, shift_y, shift_x, shift_y],
                axis=0
            )

            shifts = tf.transpose(shifts)
            # Shifts now is a (H x W, 4) Tensor

            # Expand dims to use broadcasting sum.
            all_anchors = (
                    np.expand_dims(self._anchor_reference, axis=0) +
                    tf.expand_dims(shifts, axis=1)
            )

            # Flatten
            all_anchors = tf.reshape(
                all_anchors, (-1, 4)
            )
            return all_anchors

    @property
    def summary(self):
        """
        Generate merged summary of all the sub-summaries used inside the
        Faster R-CNN network.
        """
        summaries = [
            tf.summary.merge_all(key='rpn'),
        ]

        summaries.append(
            tf.summary.merge_all(key=self._losses_collections[0])
        )

        if self._with_rcnn:
            summaries.append(tf.summary.merge_all(key='rcnn'))

        return tf.summary.merge(summaries)

    @property
    def vars_summary(self):
        return {
            key: tf.summary.merge_all(key=collection)
            for key, collections in VAR_LOG_LEVELS.items()
            for collection in collections
        }

    def get_trainable_vars(self):
        """Get trainable vars included in the module.
        """
        trainable_vars = snt.get_variables_in_module(self)
        if self._config.model.base_network.trainable:
            pretrained_trainable_vars = self.base_network.get_trainable_vars()
            if len(pretrained_trainable_vars):
                tf.logging.info(
                    'Training {} vars from pretrained module; '
                    'from "{}" to "{}".'.format(
                        len(pretrained_trainable_vars),
                        pretrained_trainable_vars[0].name,
                        pretrained_trainable_vars[-1].name,
                    )
                )
            else:
                tf.logging.info('No vars from pretrained module to train.')
            trainable_vars += pretrained_trainable_vars
        else:
            tf.logging.info('Not training variables from pretrained module')

        return trainable_vars

    def get_base_network_checkpoint_vars(self):
        return self.base_network.get_base_network_checkpoint_vars()

    def get_checkpoint_file(self):
        return self.base_network.get_checkpoint_file()
