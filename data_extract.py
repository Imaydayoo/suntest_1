import re

import pandas as pd


def getMAC():
    df = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')

    # print(type(df['专科检查'][12]))
    print(len(df))
    ans = []
    pattern = re.compile(r'腹围:?(\d+\.?\d*)c+m?')
    # 25634
    for i in range(25349):
        if type(df['专科检查'][i]) is not str or len(df['专科检查'][i]) == 0:
            ans.append('')
        else:
            temp = pattern.findall(df['专科检查'][i])
            if len(temp) > 0:
                ans.append(temp[0])
            else:
                ans.append('')
    # print(ans)
    # df.insert(25, '母体腹围cm', ans)
    df['母体腹围cm'] = ans
    df.to_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')


def getfetalheight():
    df = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')

    # print(type(df['专科检查'][12]))
    print(len(df))
    ans = []
    pattern = re.compile(r'宫高:?(\d+\.?\d*)c?m?')
    # (25349, 23)
    for i in range(25349):
        # if '宫高' in df['专科检查'][i]:
        if type(df['专科检查'][i]) is not str or len(df['专科检查'][i]) == 0:
            ans.append('')
        else:
            temp = pattern.findall(df['专科检查'][i])
            if len(temp) > 0:
                ans.append(temp[0])
            else:
                ans.append('')
    # print(ans)
    # df.insert(23, '宫高cm', ans)
    df['宫高cm'] = ans
    df.to_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')


def getbpd():
    df = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')

    # print(type(df['专科检查'][12]))
    print(len(df))
    ans = []
    pattern = re.compile(r'双顶径:?(\d+\.?\d*)')
    pattern2 = re.compile(r'BPD:?(\d+\.?\d*)')
    # 25634
    for i in range(25349):
        if type(df['专科检查'][i]) is not str or len(df['专科检查'][i]) == 0:
            ans.append('')
        else:
            temp = pattern.findall(df['专科检查'][i])
            if len(temp) > 0:
                ans.append(temp[0])
            else:
                temp2 = pattern2.findall(df['专科检查'][i])
                if len(temp2) > 0:
                    ans.append(temp2[0])
                else:
                    ans.append('')
    # print(ans[111])
    # df.insert(26, '双顶径mm', ans)
    df['双顶径mm'] = ans
    df.to_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')


def getfac():
    df = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')

    # print(type(df['专科检查'][12]))
    print(len(df))
    ans = []
    pattern = re.compile(r'腹围:?(\d+\.?\d*)mm?')
    pattern2 = re.compile(r'AC:?(\d+\.?\d*)mm?')
    # 25349
    for i in range(25349):
        if type(df['专科检查'][i]) is not str or len(df['专科检查'][i]) == 0:
            ans.append('')
        else:
            temp = pattern.findall(df['专科检查'][i])
            if len(temp) > 0:
                ans.append(temp[0])
            else:
                temp2 = pattern2.findall(df['专科检查'][i])
                if len(temp2) > 0:
                    ans.append(temp2[0])
                else:
                    ans.append('')
    # print(ans)
    # df.insert(26, '双顶径mm', ans)
    df['胎儿腹围mm'] = ans
    df.to_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')


def getfhc():
    df = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')

    # print(type(df['专科检查'][12]))
    print(len(df))
    ans = []
    pattern = re.compile(r'头围:?(\d+\.?\d*)')
    pattern2 = re.compile(r'HC:?(\d+\.?\d*)')
    # 25634
    for i in range(25349):
        if type(df['专科检查'][i]) is not str or len(df['专科检查'][i]) == 0:
            ans.append('')
        else:
            temp = pattern.findall(df['专科检查'][i])
            if len(temp) > 0:
                ans.append(temp[0])
            else:
                temp2 = pattern2.findall(df['专科检查'][i])
                if len(temp2) > 0:
                    ans.append(temp2[0])
                else:
                    ans.append('')
    # print(ans)
    # df.insert(26, '双顶径mm', ans)
    df['头围mm'] = ans
    df.to_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')


def getfl():
    df = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')

    # print(type(df['专科检查'][12]))
    print(len(df))
    ans = []
    pattern = re.compile(r'股骨长?:?(\d+\.?\d*)')
    pattern2 = re.compile(r'FL:?(\d+\.?\d*)')
    # 25634
    for i in range(25349):
        if type(df['专科检查'][i]) is not str or len(df['专科检查'][i]) == 0:
            ans.append('')
        else:
            temp = pattern.findall(df['专科检查'][i])
            if len(temp) > 0:
                ans.append(temp[0])
            else:
                temp2 = pattern2.findall(df['专科检查'][i])
                if len(temp2) > 0:
                    ans.append(temp2[0])
                else:
                    ans.append('')
    # print(ans)
    # df.insert(26, '双顶径mm', ans)
    df['股骨长mm'] = ans
    df.to_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')


def getad():
    df = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')

    # print(type(df['专科检查'][12]))
    print(len(df))
    ans = []
    pattern = re.compile(r'前后径:?(\d+\.?\d*)')
    pattern2 = re.compile(r'最大径线:?(\d+\.?\d*)')
    # 25634
    for i in range(25349):
        if type(df['专科检查'][i]) is not str or len(df['专科检查'][i]) == 0:
            ans.append('')
        else:
            temp = pattern.findall(df['专科检查'][i])
            if len(temp) > 0:
                ans.append(temp[0])
            else:
                temp2 = pattern2.findall(df['专科检查'][i])
                if len(temp2) > 0:
                    ans.append(temp2[0])
                else:
                    ans.append('')
    # print(ans)
    # df.insert(26, '双顶径mm', ans)
    df['最大前后径mm'] = ans
    df.to_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')


def getGA():
    df = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')

    # print(type(df['专科检查'][12]))
    print(len(df))
    ans = []
    pattern_1 = re.compile(r'(\d+)±?\+?\d?周')
    pattern_2 = re.compile(r'足月')
    # 25346
    for i in range(25346):
        if type(df['专科检查'][i]) is not str or len(df['专科检查'][i]) == 0:
            ans.append('')
        else:
            temp1 = pattern_1.findall(df['专科检查'][i])
            if len(temp1) > 0:
                ans.append(temp1[0])
            else:
                temp2 = pattern_2.findall(df['专科检查'][i])
                ans.append('37.55')
    # print(ans)
    df['孕周week'] = ans
    df.to_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')


def getgdm():
    df = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')

    # print(type(df['专科检查'][12]))
    print(len(df))
    ans = []
    pattern_1 = re.compile(r'糖尿病')
    # 25346
    for i in range(len(df)):
        if type(df['出院诊断'][i]) is not str or len(df['出院诊断'][i]) == 0:
            ans.append(0)
        else:
            temp1 = pattern_1.findall(df['出院诊断'][i])
            if len(temp1) > 0:
                ans.append(1)
            else:
                ans.append(0)
    # print(ans)
    df['gdm'] = ans
    df.to_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')


def getgf():
    df = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')

    # print(type(df['专科检查'][12]))
    print(len(df))
    ans = []
    pattern_1 = re.compile(r'肥胖')
    # 25346
    for i in range(len(df)):
        if type(df['出院诊断'][i]) is not str or len(df['出院诊断'][i]) == 0:
            ans.append(0)
        else:
            temp1 = pattern_1.findall(df['出院诊断'][i])
            if len(temp1) > 0:
                ans.append(1)
            else:
                ans.append(0)
    # print(ans)
    df['gf'] = ans
    df.to_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')


def get_height():
    df = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')

    # print(type(df['专科检查'][12]))
    print(len(df))
    ans = []
    pattern = re.compile(r'身高:?(\d+\.?\d*)cm?')
    # 25634
    count1 = 0
    for i in range(len(df)):
        if type(df['体格检查内容'][i]) is not str or len(df['体格检查内容'][i]) == 0:
            ans.append('')
        else:
            temp = pattern.findall(df['体格检查内容'][i])
            if len(temp) > 0:

                ans.append(temp[0])
            else:
                ans.append('')
    # print(count1)
    df['身高cm'] = ans
    df.to_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')


def get_weight():
    df = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')

    # print(type(df['专科检查'][12]))
    print(len(df))
    ans = []
    pattern = re.compile(r'体重:?(\d+\.?\d*)kg?')
    # 25634
    count1 = 0
    for i in range(len(df)):
        if type(df['体格检查内容'][i]) is not str or len(df['体格检查内容'][i]) == 0:
            ans.append('')
        else:
            temp = pattern.findall(df['体格检查内容'][i])
            if len(temp) > 0:
                count1 += 1
                ans.append(temp[0])
            else:
                ans.append('')
    # print(count1)
    df['weight'] = ans
    df.to_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')


def get_fp():
    df = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')

    # print(type(df['专科检查'][12]))
    print(len(df))
    ans = []
    count1 = 0
    for i in range(len(df)):
        if type(df['初步诊断'][i]) is not str or len(df['初步诊断'][i]) == 0:
            ans.append('2')
        else:
            if '头位' in df['初步诊断'][i]:
                ans.append('1')
            elif '臀' in df['初步诊断'][i] or '横位' in df['初步诊断'][i]:
                ans.append('0')
            else:
                ans.append('2')
        if ans[i] == '2':
            if '头位' in df['专科检查'][i] or '先露:头' in df['专科检查'][i] or '先露头' in df['专科检查'][i]:
                ans[i] = '1'
            elif '臀' in df['专科检查'][i] or '横位' in df['专科检查'][i]:
                ans[i] = '0'
            else:
                count1 += 1

    # print(count1)
    df['fp'] = ans
    df.to_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')


# df = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/test2.csv')
# print(df.shape)

# (25358, 23)
# getfetalheight()
# getMAC()
# getbpd()
# getfac()
# getfhc()
# getfl()
# getad()

# pattern2 = re.compile(r'BPD:?(\d+\.?\d*)')
# print(pattern2.findall('BPD:232.2'))
# getGA()
# getgdm()
# getgf()
# get_height()
# get_weight()
get_fp()

