# Detector-in-Detector
  Unofficial Implementation of Detector-in-Detector: Multi-Level Analysis for Human-Parts 
in simplify form.<br>

  This project is based on tensorflow object detection framework luminoth, 
and overload the faster-rcnn interface to implement detector-in-detector in 
simplify form.<br>

  Paper Link: https://arxiv.org/abs/1902.07017<br>

Install Steps：<br>
  1\ install luminoth according to https://luminoth.readthedocs.io/en/latest/<br>
  2\ copy following files in this project into your luminoth path:<br>

train.py<br>
models/fasterrcnn/__init__.py<br>
models/fasterrcnn/detector_in_detector.py<br>
models/fasterrcnn/part_detector.py<br>
models/fasterrcnn/part_detector_rcnn.py<br>
models/fasterrcnn/rcnn_overload.py<br>
models/fasterrcnn/rcnn_proposal_overload.py<br>
models/fasterrcnn/part_detector_roi_pool.py<br>

predict.py<br>
utils/detector_in_detector_predicting.py<br>

Train this model:<br>
  1\ download paper dataset according to<br>
  https://github.com/xiaojie1017/Human-Parts<br>
  2\ run dataset_script/preprocess_dataset.py to generate luminoth<br>
  friendly csv annotation file<br>
  3\ generate tfrecord file with luminoth dataset api<br>
  4\ use luminoth train -c dataset_script/start.yml to train <br>
  5\ use tensorboard to valid your net.<br>
  6\ use luminoth predict -c dataset_script/start.yml to train<br>

how to debug<br>
  train time:<br>
  edit utils/hooks/image_vis_hook.py and retrieve in after_run <br>

  predict time:<br>
  edit utils/detector_in_detector_predicting.py<br>

some difference with luminoth<br>
  this is only a edit version of luminoth FasterRCNN<br>
  the only difference between this yml with ori <br>
  faster-rcnn yml is to set <br>
  main_label_index to differciate body (main_part) bboxes <br>
  and part bboxes<br>

some difference with paper<br>
  paper use ground truth person proposals as padding to <br>
  early training process, here is replaced by random samping <br>
  and prob threshold filter. You can try to increase the <br>
  padding num (which is 7) to approximate it.  <br>
  Or you can switch it by hand.<br>

some drawbacks may require improved<br>
  for the usage of filter script, the data_augmentation <br>
  config is limited to flip and random noise add action, <br>
  some patch action may cause main and part overlap<br>
  be zero, which is harmful for training.<br>
  This is related with the construction between Body Detector <br>
  and Part Detector,If you can accumulate samples and perform <br>
  train in this fasion, you can tackle this problem.<br>

  lumi faster-rcnn only support batch-size 1, limit the <br>
  train efficiency.<br>

  Not overload lumi eval and web server interface <br>
  in this version. (may be simple)<br>

More info can be seen in my chiness blog: <br>
