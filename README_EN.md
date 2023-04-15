<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">Detector-in-Detector</h3>

  <p align="center">
  Unofficial Implementation of Detector-in-Detector: Multi-Level Analysis for Human-Parts
in simplify form.
    <br />
  </p>
</p>

[中文简介](README.md)

## Brief introduction
This project is based on tensorflow object detection framework luminoth,
and overload the faster-rcnn interface to implement detector-in-detector in
simplify form.<br>

Paper Link: https://arxiv.org/abs/1902.07017<br>

## Installtion
* 1 install luminoth according to https://github.com/tryolabs/luminoth <br>
* 2 copy following files in this project into your luminoth path:<br>

```bash
train.py
models/fasterrcnn/__init__.py
models/fasterrcnn/detector_in_detector.py
models/fasterrcnn/part_detector.py
models/fasterrcnn/part_detector_rcnn.py
models/fasterrcnn/rcnn_overload.py
models/fasterrcnn/rcnn_proposal_overload.py
models/fasterrcnn/part_detector_roi_pool.py
```

```bash
predict.py
utils/detector_in_detector_predicting.py
```

## Training steps
* 1 download paper dataset according to<br>
 https://github.com/xiaojie1017/Human-Parts<br>
* 2 run dataset_script/preprocess_dataset.py to generate luminoth<br>
 friendly csv annotation file<br>
* 3 generate tfrecord file with luminoth dataset api<br>
* 4 use luminoth train -c dataset_script/start.yml to train <br>
* 5 use tensorboard to valid your net.<br>
* 6 use luminoth predict -c dataset_script/start.yml to predict <br>

## Details
### How to debug
 <b>In train step:</b><br>
edit utils/hooks/image_vis_hook.py and retrieve in after_run <br>

 <b>In predict step:</b><br>
edit utils/detector_in_detector_predicting.py<br>

### Some difference with luminoth
This is only a edit version of luminoth FasterRCNN.
  the only difference between this yml with ori.
  faster-rcnn yml is to set
  main_label_index to differciate body (main_part) bboxes
  and part bboxes<br>

### Some difference with paper
Paper use ground truth person proposals as padding to
  early training process, here is replaced by random samping
  and prob threshold filter. You can try to increase the
  padding num (which is 7) to approximate it.  
  Or you can switch it by hand.

### Some drawbacks may require improved
For the usage of filter script, the data_augmentation
  config is limited to flip and random noise add action,
  some patch action may cause main and part overlap
  be zero, which is harmful for training.<br>
  This is related with the construction between Body Detector
  and Part Detector,If you can accumulate samples and perform
  train in this fasion, you can tackle this problem.<br>

Lumi faster-rcnn only support batch-size 1, limit the
  train efficiency.<br>

Not overload lumi eval and web server interface
  in this version. (may be simple)<br>


<!-- CONTACT -->
## Contact

<!--
Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com
-->
svjack - svjackbt@gmail.com - ehangzhou@outlook.com

<!--
Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)
-->
Project Link:[https://github.com/svjack/Detector-in-Detector](https://github.com/svjack/Detector-in-Detector)
