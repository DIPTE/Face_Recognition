# Face_Recognition
Chapter5 - Homework - loss_function&amp;Deep_Metric_Learning - shenlanxueyuan

## Dataset find it in GITHUB Share
### CASIAWebFace:

(https://drive.google.com/file/d/1wJC2aPA4AC0rI-tAL2BFs2M8vfcpX-w6/view?usp=sharing)

unzip casia-maxpy-clean.zip

cd casia-maxpy-clean

zip -F CASIA-maxpy-clean.zip --out CASIA-maxpy-clean_fix.zip

unzip CASIA-maxpy-clean_fix.zip

### LFW:

(https://pan.baidu.com/s/1Rue4FBmGvdGMPkyy2ZqcdQ)

## Homework Result

|      SEResNet18       |LFW（20epoch、batchsize=256）|       SEResNet34      |LFW（20epoch、batchsize=128）|
|:---------------------:|:---------------------------:|:---------------------:|:---------------------------:|
|       Softmax         | 0.851                       |       Softmax         |0.8578333333333333           |
|       NormFace        | 0.8428333333333334          |       NormFace        |0.8470000000000001           |
|      SpereFace        | 0.8474999999999999          |      SpereFace        |0.8651666666666665           |
|       CosFace         | 0.8488333333333333          |       CosFace         |0.8504999999999999           |
|       ArcFace         | 0.8456666666666667          |       ArcFace         |0.755                        |
|   OHEM & NormFace     | 0.8456666666666667          |   OHEM & NormFace     |0.8485000000000001           |
|FocalLoss & NormFace   | 0.8396666666666667          |FocalLoss & NormFace   |0.8504999999999999           |

notes:Train from scratch and run 20 epochs @ Tesla P100 16G

|      SEResNet18       |LFW                          |       SEResNet34      | LFW                                          |
|:---------------------:|:---------------------------:|:---------------------:|:--------------------------------------------:|
|Contrastive（Scratch） |0.6135\20epoch\batchsize=128 | Contrastive（Scratch）|        \                                     |
|  Triplet（Scratch）   |0.7968\4epoch\batchsize=64   |  Triplet（Scratch）   |0.8265\4epoch\batchsize=256\Quadro RTX8000 48G|
|Contrastive（Finetune）|0.6371666666666667\20epoch\batchsize=128|Contrastive（Finetune）|        \                                     |
| Triplet（Finetune）   |     \                       | Triplet（Finetune）   |        \                                     |
