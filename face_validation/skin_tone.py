

import openpyxl
import pandas as pd
import cv2
import numpy as np
import math

file = "data/Skintone Shades (2).xlsx"
data = pd.ExcelFile(file)
print(data.sheet_names) #this returns the all the sheets in the excel file

df = data.parse('Sheet1')


ps = openpyxl.load_workbook('data/Skintone Shades (2).xlsx')

sheet = ps['Sheet1']

sheet.max_row

dict_temp = {}
list_rgb = []
list_shade = []
for row in range(2, sheet.max_row + 1):
  shade = sheet["A"+str(row)].value
  rgb = sheet["B"+str(row)].value.replace("\xa0", " ")
  list_rgb.append(rgb)
  list_shade.append(shade)


new_list = []
for j in range(0,len(list_rgb)):
  test_list = list_rgb[j].split()
  for i in range(0, len(test_list)):
    test_list[i] = int(test_list[i])
  new_list.append(test_list)


for data in range(len(list_shade)):
  dict_temp[list_shade[data]] = new_list[data]
print(dict_temp)

new = []
myimg = cv2.imread('/content/unnamed2.jpg', cv2.IMREAD_COLOR)
channels = cv2.mean(myimg)
observation = np.array((channels[2], channels[1], channels[0])).tolist()
for i in observation:
  new.append(math.trunc(i))
print(new)

def find_closest(new):
  r, g, b = new
  color_diffs = []
  for color in new_list:
      cr, cg, cb = color
      color_diff = math.sqrt(abs(r - cr)**2 + abs(g - cg)**2 + abs(b - cb)**2)
      color_diffs.append((color_diff, color))
  return min(color_diffs)[1]

values = dict_temp.items()
keys = dict_temp.keys()
for key, value in dict_temp.items():
  if value == new:
    print(key)
  else:
    new_out = find_closest(new)
    if value == new_out:
      # print(value)
      print(key)
