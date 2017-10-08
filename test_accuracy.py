
file_path = 'D:/visionhack/trainset/trainset/train.txt'

answers = {}
assoc = {
    0: 'bridge',
    1: 'city',
    2: 'city_2',
    3: 'road_bump',
    4: 'screen_wipers',
    5: 'zebra'
}

# fill dict
with open(file_path) as f:
    content = f.readlines()
    for line in content:
        row = line.split(' ')
        values = list(row[1])
        answers[row[0]] = values


class TestInfo:
    data = {}

    def __init__(self, data):
        self.data = data


class ExInfo:
    positive = 0
    false_positive = 0
    missed = 0


def check_answer(answer_file_path):
    with open(answer_file_path) as a_f:
        answer_content = a_f.readlines()

        dict = {
            'bridge': ExInfo(),
            'city': ExInfo(),
            'city_2': ExInfo(),
            'road_bump': ExInfo(),
            'screen_wipers': ExInfo(),
            'zebra': ExInfo()
        }
        test_info = TestInfo(dict)

        for _s in answer_content:
            _line = _s.split(' ')
            vals = answers[_line[0]]

            for ind, _val in enumerate(list(_line[1].rstrip())):
                if _val == vals[ind]:
                    if _val == '1':
                        test_info.data[assoc[ind]].positive += 1
                elif _val == '0':
                    test_info.data[assoc[ind]].missed += 1
                else:
                    test_info.data[assoc[ind]].false_positive += 1

    return test_info


def print_test_info(test_info):
    for key in test_info.data.keys():
        print("_____" + key + "_____")
        print("Positive: " + str(test_info.data[key].positive))
        print("False_positive: " + str(test_info.data[key].false_positive))
        print("Missed: " + str(test_info.data[key].missed))

test_i = check_answer('answer.txt')
print_test_info(test_i)
