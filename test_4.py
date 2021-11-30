import numpy as np
import pandas as pd


def delete_lack_char():
    df = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')
    print(len(df))
    n = len(df)
    count = 0

    ans = []
    for i in range(n):
        # if float(df['婴儿体重'][i]) <= 2.50 and float(df['孕周week'][i]) >= 35:
        # if float(df['孕周week'][i]) >= 37:
        #     count += 1
        # if df['住院号'][i] == df['住院号'][i + 1]:
        #     count += 1
        #     print(i)
        # if df['宫高cm'][i] > 0 and df['母体腹围cm'][i] > 0:
        #     count += 1
        if np.isnan(df['FH'][i]) or np.isnan(df['MAC'][i]) or np.isnan(df['BPD'][i]) or \
                np.isnan(df['FAC'][i]) or np.isnan(df['HC'][i]) or np.isnan(df['FL'][i]) or \
                np.isnan(df['AD'][i]) or np.isnan(df['GA'][i]) or np.isnan(df['height'][i]) \
                or np.isnan(df['weight'][i]) or df['fp'][i] == 2:
            count += 1
            ans.append(i)
    print(count)
    print(ans)
    df.drop(ans, inplace=True)
    print(df.shape)
    df.to_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')


def delete_fh():
    df = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')
    print(len(df))
    n = len(df)
    count = 0

    ans = []
    for i in range(n):
        if df['宫高cm'][i] > 50 or df['宫高cm'][i] < 20:
            count += 1
            ans.append(i)
    print(count)
    print(ans)
    df.drop(ans, inplace=True)
    print(df.shape)
    df.to_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')


def delete_mac():
    df = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')
    print(len(df))
    n = len(df)
    count = 0

    ans = []
    for i in range(n):
        if df['母体腹围cm'][i] > 150 or df['母体腹围cm'][i] < 50:
            count += 1
            ans.append(i)
    print(count)
    print(ans)
    df.drop(ans, inplace=True)
    print(len(df))
    df.to_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')


def delete_bpd():
    df = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')
    print(len(df))
    n = len(df)
    count = 0

    ans = []
    for i in range(n):
        if df['双顶径mm'][i] > 110 or df['双顶径mm'][i] < 50:
            count += 1
            ans.append(i)
    print(count)
    print(ans)
    df.drop(ans, inplace=True)
    print(len(df))
    df.to_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')


def delete_fac():
    df = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')
    print(len(df))
    n = len(df)
    count = 0

    ans = []
    for i in range(n):
        if df['胎儿腹围mm'][i] > 500 or df['胎儿腹围mm'][i] < 200:
            count += 1
            ans.append(i)
    print(count)
    print(ans)
    df.drop(ans, inplace=True)
    print(len(df))
    df.to_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')


def delete_fhc():
    df = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')
    print(len(df))
    n = len(df)
    count = 0

    ans = []
    for i in range(n):
        if df['头围mm'][i] > 500 or df['头围mm'][i] < 200:
            count += 1
            ans.append(i)
    print(count)
    print(ans)
    df.drop(ans, inplace=True)
    print(len(df))
    df.to_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')


def delete_fl():
    df = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')
    print(len(df))
    n = len(df)
    count = 0

    ans = []
    for i in range(n):
        if df['股骨长mm'][i] > 100 or df['股骨长mm'][i] < 30:
            count += 1
            ans.append(i)
    print(count)
    print(ans)
    df.drop(ans, inplace=True)
    print(len(df))
    df.to_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')


def delete_ad():
    df = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')
    print(len(df))
    n = len(df)
    count = 0

    ans = []
    for i in range(n):
        if df['最大前后径mm'][i] > 150 or df['最大前后径mm'][i] < 20:
            count += 1
            ans.append(i)
    print(count)
    print(ans)
    df.drop(ans, inplace=True)
    print(len(df))
    df.to_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')


def delete_GA():
    df = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')
    print(len(df))
    n = len(df)
    count = 0

    ans = []
    for i in range(n):
        if df['孕周week'][i] < 35:
            count += 1
            ans.append(i)
    print(count)
    print(ans)
    df.drop(ans, inplace=True)
    print(len(df))
    df.to_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')


def delete_height():
    df = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')
    print(len(df))
    n = len(df)
    count = 0

    ans = []
    for i in range(n):
        if df['height'][i] < 100.0:
            count += 1
            ans.append(i)
    print(count)
    # print(ans)
    df.drop(ans, inplace=True)
    print(len(df))
    df.to_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')


def delete_weight():
    df = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')
    print(len(df))
    n = len(df)
    count = 0

    ans = []
    for i in range(n):
        if df['weight'][i] < 30 or df['weight'][i] > 120:
                count += 1
                ans.append(i)
    print(count)
    print(ans)
    df.drop(ans, inplace=True)
    print(len(df))
    df.to_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')


# delete_lack_char()
# delete_fh()
# delete_mac()
# delete_GA()
# delete_bpd()
# delete_fhc()
# delete_fl()
# delete_ad()
# delete_height()
delete_weight()
