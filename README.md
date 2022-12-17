# Log-Anomaly-Detection
Based on the implementation of Deeplog, we introduced Transformer to improve the performance and proposed TransLog.

## Overall Architecture
![TransLog网络架构](figures/architecture.png)

## Experimental Results
### Training Loss 
![Loss](figures/train_loss.png)

### Validation Loss
![Loss](figures/valid_loss.png)

### Number of Parameters
![Loss](figures/num.png)

## Usage
1. Run parse_log.py to parse the raw log file.
2. Run train.py to train the model. The architecture of model is
defined in models/model_collects.py, and you can change it for
creating costum models. 
3. Experiment is realized in work.py, including the training for different
comparsion netowrks and varients of TransLog, the performance evaluation,
and some others.
