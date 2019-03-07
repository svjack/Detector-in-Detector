# Detector-in-Detector
Unofficial Implementation of Detector-in-Detector: Multi-Level Analysis for Human-Parts in simplify form.

This project is based on tensorflow object detection framework luminoth, 
and overload the faster-rcnn interface to implement detector-in-detector in 
simplify form.

Paper Link: https://arxiv.org/abs/1902.07017

Install Steps：
1\ install luminoth according to https://luminoth.readthedocs.io/en/latest/
2\ copy following files in this project into your luminoth path:

train.py
models/fasterrcnn/__init__.py
models/fasterrcnn/detector_in_detector.py
models/fasterrcnn/part_detector.py
models/fasterrcnn/part_detector_rcnn.py
models/fasterrcnn/rcnn_overload.py
models/fasterrcnn/rcnn_proposal_overload.py
models/fasterrcnn/part_detector_roi_pool.py

predict.py
utils/detector_in_detector_predicting.py

Train this model:
1\ download paper dataset according to https://github.com/xiaojie1017/Human-Parts
2\ run dataset_script/preprocess_dataset.py to generate luminoth friendly csv annotation file
3\ generate tfrecord file with luminoth dataset api
4\ use luminoth train -c dataset_script/start.yml to train 
5\ use tensorboard to valid your net.
6\ use luminoth predict -c dataset_script/start.yml to train

how to debug ?????
  train time:
  edit utils/hooks/image_vis_hook.py and retrieve in after_run 

  predict time:
  edit utils/detector_in_detector_predicting.py

some difference with luminoth
  this is only a edit version of luminoth FasterRCNN
  the only difference between this yml with ori faster-rcnn yml is to set   main_label_index to differciate body (main_part) bboxes and part bboxes

some difference with paper
  paper use ground truth person proposals as padding to early training process, 
  here is replaced by random samping and prob threshold filter. You can try to increase the padding num (which is 7) to approximate it.  
  Or you can switch it by hand.

some drawbacks may require improved
  for the usage of filter script, the data_augmentation config is limited to
   flip and random noise add action, some patch action may cause main and part overlap
  be zero, which is harmful for training.
  This is related with the construction between Body Detector and Part Detector,
  If you can accumulate samples and perform train in this fasion, you can tackle this
  problem.

  lumi faster-rcnn only support batch-size 1, limit the train efficiency.

  Not overload lumi eval and web server interface in this version. (may be simple)

More info can be seen in my chiness blog: 
