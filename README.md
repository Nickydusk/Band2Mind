# Band2Mind

### 已完成

- 数据生成：`data_generate.py`
  - 样本生成完毕，在`/data/train/`，每一个csv文件为一个样例
  - 文件名train_xxx_label，最后一位label表示正例/负例
  - 生成数据每一个包含2000多行，20列，第一列为每隔一分钟采样一次的时间（总48小时），后19列为正态分布随机数据
- 特征提取：`feature_extraction.py`
  - 提取后每一行为一个样本，每一列为一个特征，最后一列为label
  - 特征有：偏度、锋度、相关性、FFT
  - 生成后保存在`/data/train_features.csv`



### 有待完成

- 特征数据预处理，标准化之类的
- 降维
- 决策树