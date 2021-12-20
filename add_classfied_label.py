import pandas as pd


def add_label():
    df = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/data.csv')
    ans = []
    # ans1 = []
    # ans2 = []
    # ans3 = []
    # print(len(df['label']))
    count = 0
    for i in range(len(df)):
        if df['label'][i] <= 2.5:
            ans.append(0)
            # ans1.append(1)
            # ans2.append(0)
            # ans3.append(0)
        elif df['label'][i] >= 4.0:
            ans.append(2)
            # ans1.append(0)
            # ans2.append(0)
            # ans3.append(1)

        else:
            ans.append(1)
            # count += 1
            # ans1.append(0)
            # ans2.append(1)
            # ans3.append(0)

    # print(ans2[0:10])
    # df['lw'] = ans1
    # df['mw'] = ans2
    # df['hw'] = ans3
    df['single_label'] = ans
    df.to_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/data.csv')


def view_data():
    df = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')
    ans = []
    count = 0
    # print(len(df['label']))
    for i in range(len(df)):
        # if 4.0 > df['婴儿体重'][i] > 2.50:
            count += 1

    print(count)
    # df.to_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/classified_data.csv')


def get_cv():
    # fetal_weight_data = pd.read_csv("data/housing.data", header=0, index_col=None, sep='\s+')
    fetal_weight_data = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/data.csv')
    print("data_shape:", fetal_weight_data.shape)

    data_sample = fetal_weight_data.iloc[:, :].values
    # data_label = fetal_weight_data.iloc[:, -1].values.reshape(-1, 1)

    mean = data_sample.mean(axis=0)
    std = data_sample.std(axis=0)
    cv = std / mean
    print(mean)
    print(std)
    print(cv)


add_label()
# view_data()
# get_cv()
