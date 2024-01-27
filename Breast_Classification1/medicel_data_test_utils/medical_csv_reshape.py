import pandas as pd

# 读取CSV文件
df = pd.read_csv('/home/tianyu/Desktop/pytorch-classification/train_phase2/DBT_phase2_train_bboxes.csv')

# 遍历指定列，并对每一行进行更改
for index, value in enumerate(df['label']):
    # 在这里进行你的更改操作，例如将每个值加上10
    df.loc[index, 'label'] = value.replace('rgb', 'grey')

# 保存修改后的CSV文件
df.to_csv('/home/tianyu/Desktop/pytorch-classification/train_phase2/DBT_phase2_train_bboxes.csv', index=False)

