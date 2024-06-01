import json
import os.path
import random
import re


def split(dir_path="."):
    all_paris = []
    all_lines = []
    scale = [6, 4, 0]  # train:test:valid

    with open(os.path.join(dir_path, "all.txt"), "r", encoding="utf-8") as f:
        lines = f.readlines()
        random.shuffle(lines)

        for line in lines:
            pair = re.split(r"\t", line.strip())

            in_seq = pair[0]
            out_seq = pair[1]

            # in_seq = "".join(re.split(r"\s+", in_seq))
            # out_seq = "".join(re.split(r"\s+", out_seq))

            all_paris.append([in_seq, out_seq])
            all_lines.append(in_seq + "\t" + out_seq)

        # 测试集大小
        train_size = int(scale[0]/sum(scale) * len(lines))
        test_size = int(scale[1]/sum(scale) * len(lines))
        # 划分数据集
        train_lines = all_lines[:train_size]
        test_lines = all_lines[train_size:test_size+train_size]
        valid_lines = all_lines[train_size+test_size:]

    print(all_lines)

    with open(os.path.join(dir_path, "train.txt"), "w", encoding="utf-8") as f:
        for line in train_lines:
            f.write(line)
            f.write("\n")

    with open(os.path.join(dir_path, "test.txt"), "w", encoding="utf-8") as f:
        for line in test_lines:
            f.write(line)
            f.write("\n")

    with open(os.path.join(dir_path, "valid.txt"), "w", encoding="utf-8") as f:
        for line in valid_lines:
            f.write(line)
            f.write("\n")


if __name__ == "__main__":
    split()
