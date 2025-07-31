import os
fns = os.listdir("tiny_local_models")
fns = [x.replace(".txt", "") for x in fns if x.endswith(".txt")]
fns = [x for x in fns if x != "models"]
fns2 = '\n'.join(fns)
print(fns2)
print(len(fns))

with open("tiny_local_models/models.txt") as fp:
    data = fp.read()
    data = data.split("\n")
    data = data[2:]
data2 = '\n'.join(data)
print(data2)
print(len(data))

no_created = sorted(set(data).difference(fns))
print(len(no_created))
no_created2 = '\n'.join(no_created)
print(no_created2)

info = []
for fn in  fns:
    fn = os.path.join("tiny_local_models", f"{fn}.txt")
    with open(fn) as fp:
        time = fp.read()
        info.append((fn, time))

info = sorted(info, key=lambda x: float(x[1]))

for (fn, time) in info:
    print(f"{fn}: {time}")
