import pandas as pd

df1 = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test_label.csv')  # 可以添加sheet名指定表
df2 = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test_info.csv')

# data1 = df1.loc[:, ['住院号', '住院序数', '婴儿体重']].values  # 0表示第一行 这里读取数据并不包含表头，要注意哦！
# data2 = df2.loc[:, ['住院号', '住院序数', '专科检查']].values


# # # 将data2的数据追加到data1上
# mergeResult = pd.merge(df1, df2, on=['住院号', '住院序数'])
# print(mergeResult)
# mergeResult.to_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/merge.csv', na_rep='NA')
print('s')


