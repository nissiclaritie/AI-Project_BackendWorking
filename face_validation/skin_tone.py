import openpyxl
import pandas as pd
import cv2
import numpy as np
import math

file = "data/Skintone Shades (2).xlsx"
data = pd.ExcelFile(file)
print(data.sheet_names)  # this returns the all the sheets in the excel file

df = data.parse('Sheet1')

ps = openpyxl.load_workbook(file)

sheet = ps['Sheet1']

dict_temp = {}
list_rgb = []
list_shade = []

for row in range(2, sheet.max_row + 1):
    shade = sheet["A" + str(row)].value
    rgb = sheet["B" + str(row)].value.replace("\xa0", " ")
    list_rgb.append(rgb)
    list_shade.append(shade)
print(list_rgb)

new_list = []
for j in range(0, len(list_rgb)):
    test_list = list_rgb[j].split()
    for i in range(0, len(test_list)):
        test_list[i] = int(test_list[i])
    new_list.append(test_list)

for data in range(len(list_shade)):
    dict_temp[list_shade[data]] = new_list[data]


def find_closest(new):
    r, g, b = new
    color_diffs = []
    for color in new_list:
        cr, cg, cb = color
        color_diff = math.sqrt(abs(r - cr) ** 2 + abs(g - cg) ** 2 + abs(b - cb) ** 2)
        color_diffs.append((color_diff, color))
    return min(color_diffs)[1]


def distance(c1, c2):
    (r1, g1, b1) = c1
    (r2, g2, b2) = c2
    return math.sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2)


def image_rgb(myimg):
    new = []
    myimg = myimg
    channels = cv2.mean(myimg)
    observation = np.array((channels[2], channels[1], channels[0])).tolist()
    for obs in observation:
        new.append(math.trunc(obs))

    values = dict_temp.items()
    keys = dict_temp.keys()
    closest_colors = None
    key_temp = []
    for key, value in dict_temp.items():
        if value == new:
            closest_colors = sorted(new_list, key=lambda color: distance(color, new))
            for obs in range(3):
                closest_color = closest_colors[obs]
                for key, value in dict_temp.items():
                    if closest_color == value and key not in key_temp:
                        key_temp.append(key)
        else:
            new_out = find_closest(new)
            if value == new_out:
                closest_colors = sorted(new_list, key=lambda color: distance(color, new))
                print(closest_colors)
            for obs in range(3):
                if closest_colors:
                    closest_color = closest_colors[obs]
                    for key, value in dict_temp.items():
                        if closest_color == value and key not in key_temp:
                            key_temp.append(key)
    return key_temp
