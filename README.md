# AI3612-CBLUE

[CBLUE: A Chinese Biomedical Language Understanding Evaluation Benchmark tasks](https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414) based on CMeEE (Chinese Medical Named Entity Recognition Dataset), SJTU AI3612 Course Project.

## About CMeEE:

- CMeEE_train.json：15000
- CMeEE_dev.json ：5000
- CMeEE_test.json ：3000

The corpus contains 938 files and 47,194 sentences. The average number of words contained per file is 2,355. The dataset contains 504 common pediatric diseases, 7,085 body parts, 12,907 clinical symptoms, and 4,354 medical procedures in total.

Detailed: https://arxiv.org/pdf/2106.08087v6.pdf

For (possible?) copyright reasons, please download CMeEE dataset from https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414,
and directly put `data` under the root of this project.

## About Bert Pretrained Model

For storage reasons, please download from https://huggingface.co/bert-base-chinese/tree/main, and directly put `bert-base-chinese` under the root of this project.

## Contributor

Ziyuan Li: Finish `metrics.py`; Add Layer Learning rate decay (todo)

Chuhang Ma: Finish `ee_data.py` and `model.py`; Add biLSTM & Loss Average.

Juntu Zhao: Finish `ee_data.py` and `model.py`; Write the course report.
