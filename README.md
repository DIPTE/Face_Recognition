# face_recognition
Chapter5 - Homework - loss_function&amp;Deep_Metric_Learning - shenlanxueyuan


# Dataset find it in GITHUB Share
## CASIAWebFace:

(https://drive.google.com/file/d/1wJC2aPA4AC0rI-tAL2BFs2M8vfcpX-w6/view?usp=sharing)

unzip casia-maxpy-clean.zip

cd casia-maxpy-clean

zip -F CASIA-maxpy-clean.zip --out CASIA-maxpy-clean_fix.zip

unzip CASIA-maxpy-clean_fix.zip

## LFW:

(https://pan.baidu.com/s/1Rue4FBmGvdGMPkyy2ZqcdQ)

## Homework Result

|      SEResNet18       |   LFW              |       SEResNet34      |   LFW              |
|:---------------------:|:------------------:|:---------------------:|:------------------:|
|       Softmax         | 0.851              |       Softmax         |                    |
|       NormFace        | 0.8428333333333334 |       NormFace        |                    |
|      SpereFace        | 0.8474999999999999 |      SpereFace        |                    |
|       CosFace         | 0.8488333333333333 |       CosFace         |                    |
|       ArcFace         | 0.8456666666666667 |       ArcFace         |                    |
|   OHEM & NormFace     | 0.8456666666666667 |   OHEM & NormFace     |                    |
|FocalLoss & NormFace   | 0.8396666666666667 |FocalLoss & NormFace   |                    |
|     Contrastive       |                    |     Contrastive       |                    |
|        Triplet        |                    |        Triplet        |                    |
| Contrastive & Finetune|                    | Contrastive & Finetune|                    |
| Triplet & Finetune    |                    | Triplet & Finetune    |                    |


|	      SEResNet18      |        LFW         |	      SEResNet34       |        LFW         |
|:---------------------:|:------------------:|:----------------------:|:------------------:|
|Contrastive（Scratch） | 0.6135             | Contrastive（Scratch） | 0.6135             |
|Triplet（Scratch）	    |                    |Triplet（Scratch）	     |                    |
|Contrastive（Finetune）|                    |Contrastive（Finetune） |                    |		
|Triplet（Finetune）	   |                    |Triplet（Finetune） 	   |                    |
