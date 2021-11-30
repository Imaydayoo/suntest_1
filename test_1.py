import xlrd

# 打开excel
# wb = xlrd.open_workbook('/Users/apple/Desktop/test_1.xlsx')
wb1 = xlrd.open_workbook('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/2015-2020产科入院记录（已标注）-1.xlsx')
wb2 = xlrd.open_workbook('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/2015-2020分娩记录单(已标注).xlsx')
# 按工作簿定位工作表
sh1 = wb1.sheet_by_name('Sheet1')
sh2 = wb2.sheet_by_name('Sheet1')
n1 = sh1.nrows
n2 = sh2.nrows
print(sh1.nrows)  # 有效数据行数
print(sh1.ncols)  # 有效数据列数
print(sh2.nrows)  # 有效数据行数
print(sh2.ncols)  # 有效数据列数
# print(sh1.col_values(0))

count = 0
slice1 = sh1.col_values(2)[3:]
slice2 = sh2.col_values(1)[1:]
slice3 = sh2.col_values(2)[1:]

# for temp in slice3:
#     if temp > 1.0:
#         count += 1


# 能找到带label的
for str1 in slice2:
    temp = str(int(str1))
    for str2 in slice1:
        if temp == str2:
            count += 1
            break


print(count)

# print(slice1[1])
# print(str(int(slice2[1])) == '336008')
# count = 0
# for i in sh1.row_values(0):
#     for j in sh2.row_values(0):
#         if i == j:
#             count += 1
#             break
# print(count)

# print(sh.cell(0, 0).value)  # 输出第一行第一列的值
# print(sh.row_values(0))  # 输出第一行的所有值
# print(sh.col_values(0))  # 输出第一列的所有值
# 将数据和标题组合成字典
# print(dict(zip(sh.row_values(0), sh.row_values(1))))
# # 遍历excel，打印所有数据
# s1 = "周"
# s2 = "足月"

# 求某个单独的特征的简单筛选
# count = 0
# for i in range(sh1.nrows):
#     if i == 0:
#         continue
#     temp = sh1.cell(i, 27).value
#     # if s1 in str(temp):
#     try:
#         if temp != '' and float(temp) >= 4.0:
#             count += 1
#     except ValueError:
#         print(temp)
# print(count)

# print(sh.row_values(i))
