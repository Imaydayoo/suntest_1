import re

import pandas as pd


def getMAC():
    df = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')

    # print(type(df['专科检查'][12]))
    print(len(df))
    ans = []
    pattern = re.compile(r'腹围:?\d+')
    # 25634
    for i in range(25634):
        if type(df['专科检查'][i]) is not str or len(df['专科检查'][i]) == 0:
            ans.append('')
        else:
            temp = pattern.findall(df['专科检查'][i])
            if len(temp) > 0:
                if temp[0][2] == ':':
                    ans.append(temp[0][3:])
                else:
                    ans.append(temp[0][2:])
            else:
                ans.append('')
    # print(ans)
    df.insert(27, '母体腹围cm', ans)
    df.to_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv', na_rep='NA')


def getfetalheight():
    df = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')

    # print(type(df['专科检查'][12]))
    print(len(df))
    ans = []
    pattern = re.compile(r'宫高:?\d+')
    for i in range(25634):
        # if '宫高' in df['专科检查'][i]:
        if type(df['专科检查'][i]) is not str or len(df['专科检查'][i]) == 0:
            ans.append('')
        else:
            temp = pattern.findall(df['专科检查'][i])
            if len(temp) > 0:
                ans.append(temp[0][2:])
            else:
                ans.append('')
    # print(ans)
    df.insert(26, '宫高cm', ans)
    df.to_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')


def getDpd():
    df = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test.csv')

    # print(type(df['专科检查'][12]))
    print(len(df))
    ans = []
    pattern = re.compile(r'双顶径:?\d+')
    # 25634
    for i in range(25634):
        if type(df['专科检查'][i]) is not str or len(df['专科检查'][i]) == 0:
            ans.append('')
        else:
            temp = pattern.findall(df['专科检查'][i])
            if len(temp) > 0:
                if temp[0][2] == ':':
                    ans.append(temp[0][4:])
                else:
                    ans.append(temp[0][3:])
            else:
                ans.append('')
    # print(ans)
    df.insert(28, '双顶径mm', ans)
    df.to_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv', na_rep='NA')


df = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')

print(df.shape)
# getfetalheight()
# getMAC()
# getDpd()
