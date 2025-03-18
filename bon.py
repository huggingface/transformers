with open(r"C:\Users\yih-d\Desktop\naturalisation-QA.txt", encoding="utf-8") as fp:
    data = fp.read()

lines = []
buf = []

for line in data.splitlines():
    if line.strip() == "":
        continue
    if line.startswith(("Q: ", "A: ")):
        buf.append(line)
    if len(buf) == 2:
        lines.append(buf)
        buf = []

for idx, buf in enumerate(lines):
    print(f"{idx + 1}".zfill(3) + ".")
    print(buf[0])
    print(buf[1])
    print("")