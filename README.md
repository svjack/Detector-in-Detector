<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">Detector-in-Detector</h3>

  <p align="center">
  “Detector-in-Detector: Multi-Level Analysis for Human-Parts” 的非官方实现：针对人体部位的多层次分析。
    <br />
  </p>
</p>

[In English](README_EN.md)

## 简要引述
该项目基于TensorFlow目标检测框架Luminoth，并重载了Faster-RCNN接口，以实现简化形式的 Detector-in-Detector。 <br>

论文链接: https://arxiv.org/abs/1902.07017<br>

## 安装
* 1 根据文档 https://luminoth.readthedocs.io/en/latest/ 安装 luminoth <br>
* 2 将下面的文件拷贝入 luminoth 源码工程文件路径:<br>

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

## 训练过程
* 1 根据下面的链接下载论文中的数据集<br>
 https://github.com/xiaojie1017/Human-Parts<br>
* 2 根据 dataset_script/preprocess_dataset.py 来生成适合 luminoth<br>
 使用的csv标注文件<br>
* 3 根据 luminoth dataset api 生成 tfrecord 格式数据文件<br>
* 4 使用 luminoth train -c dataset_script/start.yml 来训练 <br>
* 5 使用 tensorboard 来观察训练过程.<br>
* 6 使用 luminoth predict -c dataset_script/start.yml 来预测 <br>

## 细节
### 如何调试源码
 <b>在训练过程中:</b><br>
编辑 utils/hooks/image_vis_hook.py 从挂钩函数 after_run 中获得结果 <br>

 <b>在预测过程中:</b><br>
编辑 utils/detector_in_detector_predicting.py<br>

### 与 luminoth 的一些区别
这只是luminoth FasterRCNN 的编辑更改版本。与原版不同之处在于将 main_label_index 设置为区分主体（main_part 的）bbox和 身体组成成分的 bbox。<br/>

### 与论文实现叙述的区别
论文中使用人物的基准特征作为 padding 来进行早期训练过程，这里则被随机采样和概率阈值过滤替换。您可以尝试增加填充数量（即7）来逼近它。或者您也可以手动切换。<br/>

### 一些可能需要更改的实现缺陷
* 1 对于过滤脚本(filter script)的使用，数据增强配置仅限于翻转和随机噪声添加动作，有些数据增强操作可能会导致主要和部分的重叠为零，这对训练是有害的。这与身体(body)检测器和部件(part)检测器之间的构造有关，如果您可以积累样本并筛选有效样本的方式进行训练，您可以解决这个问题。<br/>

* 2 luminoth faster-rcnn 只支持 batch-size 1, 限制了训练效率.<br>

* 3 没有重载 luminoth eval 和 网络服务部署接口 (较为简单)<br>


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
