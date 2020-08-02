# face_recognition
Chapter5 - Homework - loss_function&amp;Deep_Metric_Learning - shenlanxueyuan

|      SEResNet18       |   LFW              |       SEResNet34       |   LFW              |
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


|	      SEResNet18      |        LFW         |	      SEResNet18      |        LFW         |
|:---------------------:|:------------------:|:----------------------:|:------------------:|
|Contrastive（Scratch） | 0.6135             | Contrastive（Scratch） | 0.6135             |
|Triplet（Scratch）	    |                    |Triplet（Scratch）	    |                    |
|Contrastive（Finetune）|                    |Contrastive（Finetune） |                    |		
|Triplet（Finetune）	  |                    |Triplet（Finetune） 	  |                    |
