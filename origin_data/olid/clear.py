import csv


def task_a(text, label):
    with open('./clear/train.tsv', 'a+', encoding="utf-8") as task_a:
        line = label + '\t' + text + '\n'
        csv.writer(task_a)
        task_a.write(line)


def task_b(text, label):
    with open('./clear/task_b_train.tsv', 'a+', encoding="utf-8") as task_b:
        line = label + '\t' + text + '\n'
        csv.writer(task_b)
        task_b.write(line)


def task_c(text, label):
    with open('./clear/task_c_train.tsv', 'a+', encoding="utf-8") as task_c:
        line = label + '\t' + text + '\n'
        csv.writer(task_c)
        task_c.write(line)


# param 存入文件 label文件 id 文本
def test_dev(file, label_file, text_id, text):
    with open(label_file, 'r', encoding="utf-8") as data:
        content = csv.reader(data, delimiter=',')
        for line in content:
            label_id = line[0]
            label = line[1]
            if text_id == label_id:
                row = label + '\t' + text + '\n'
                with open("clear/"+file, 'a+', encoding="utf-8") as task:
                    csv.writer(task)
                    task.write(row)


def main():
    with open('olid-training-v1.0.tsv', 'r', encoding="utf-8") as origin_data:
        content = csv.reader(origin_data, delimiter='\t')
        for line in content:
            text = line[1]
            sub_a = line[2]  # 1为冒犯性，0为不冒犯性
            sub_b = line[3]  # 只有task1为1时，才有task2,只有task2为TIN（有目标时，才有3）
            sub_c = line[4]
            task_a(text, sub_a)

            if sub_a == "1":
                task_b(text, sub_b)

            if sub_b == "TIN":
                task_c(text, sub_c)


#  验证集
# @param 文本文件 标签文件 存入文件
def main_dev(text_file, label_file, dev_file):
    with open(text_file, 'r', encoding="utf-8") as origin_data:
        content = csv.reader(origin_data, delimiter='\t')
        for line in content:
            text_id = line[0]
            text = line[1]
            test_dev(dev_file, label_file, text_id, text)


if __name__ == "__main__":
    # main()
    main_dev("testset-levelc.tsv", "labels-levelc.csv", "task_c_dev.tsv")
