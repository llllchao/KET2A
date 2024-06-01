import re

pairs = []
in_entities = set()
out_entities = set()


with open("raw.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        arr = re.split(r"\s+", line.strip())

        if len(arr) == 1:
            continue

        in_s = " ".join(arr[:5])
        out_s = " ".join(arr[5:])
        pairs.append([in_s, out_s])

for in_s, out_s in pairs:
    in_e = re.split(r"\s+", in_s)
    for e in in_e:
        in_entities.add(e)
    out_e = re.split(r"\s+", out_s)
    for e in out_e:
        out_entities.add(e)

print(in_entities)
print(out_entities)
print(pairs)

with open("seq2seq/out-entities.txt", "w", encoding="utf-8") as f:
    for i, e in enumerate(out_entities):
        f.write(str(i))
        f.write("\t")
        f.write(e)
        f.write("\n")

with open("seq2seq/in-entities.txt", "w", encoding="utf-8") as f:
    for i, e in enumerate(in_entities):
        f.write(str(i))
        f.write("\t")
        f.write(e)
        f.write("\n")

with open("seq2seq/all.txt", "w", encoding="utf-8") as f:
    for in_s, out_s in pairs:
        f.write(in_s)
        f.write("\t")
        f.write(out_s)
        f.write("\n")

