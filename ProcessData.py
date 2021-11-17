import sys
import os

import pandas as pd

def process(path):
    data = pd.read_excel(path)

    data['Chỗ ở'] = data['Chỗ ở'].replace('phòng trọ', 0)
    data['Chỗ ở'] = data['Chỗ ở'].replace('chung cư', 1)
    data['Chỗ ở'] = data['Chỗ ở'].replace('nhà thuê', 2)
    data['Chỗ ở'] = data['Chỗ ở'].replace('nhà riêng', 3)

    data['Lương'] = data['Lương'].replace('< 10', 0)
    data['Lương'] = data['Lương'].replace('10-15', 1)
    data['Lương'] = data['Lương'].replace('15-20', 2)
    data['Lương'] = data['Lương'].replace('20-30', 3)
    data['Lương'] = data['Lương'].replace('30-40', 4)
    data['Lương'] = data['Lương'].replace('> 40', 5)

    data['Công việc'] = data['Công việc'].replace('TX con người', 5)
    data['Công việc'] = data['Công việc'].replace('kỹ thuật', 4)
    data['Công việc'] = data['Công việc'].replace('hành chính', 3)
    data['Công việc'] = data['Công việc'].replace('văn hoá nghệ thuật', 2)
    data['Công việc'] = data['Công việc'].replace('công nhân', 1)
    data['Công việc'] = data['Công việc'].replace('TX tự nhiên', 0)

    data['Bằng cấp'] = data['Bằng cấp'].replace('> đại học',3)
    data['Bằng cấp'] = data['Bằng cấp'].replace('đại học',2)
    data['Bằng cấp'] = data['Bằng cấp'].replace('cao đẳng',1)
    data['Bằng cấp'] = data['Bằng cấp'].replace('phổ thông',0)

    data['Tuổi'] = data['Tuổi'].replace('< 23', 0)
    data['Tuổi'] = data['Tuổi'].replace('23-30', 1)
    data['Tuổi'] = data['Tuổi'].replace('30-40', 2)
    data['Tuổi'] = data['Tuổi'].replace('40-50', 3)
    data['Tuổi'] = data['Tuổi'].replace('> 50', 4)

    data['Thẻ tín dụng'] = data['Thẻ tín dụng'].replace('có', 1)
    data['Thẻ tín dụng'] = data['Thẻ tín dụng'].replace('không', 0)


    data['Phương tiện'] = data['Phương tiện'].replace('phương tiện công cộng', 0)
    data['Phương tiện'] = data['Phương tiện'].replace('xe máy', 1)
    data['Phương tiện'] = data['Phương tiện'].replace('ô tô', 2)

    data['Phân mức'] = data['Phân mức'].replace('bình dân', 0)
    data['Phân mức'] = data['Phân mức'].replace('trung lưu', 1)
    data['Phân mức'] = data['Phân mức'].replace('thượng lưu', 2)

    return data

