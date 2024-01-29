import os

tests = os.getcwd()
model_tests = os.listdir(os.path.join(tests, "models"))
d1 = sorted(list(filter(os.path.isdir, os.listdir(tests))))
d2 = sorted(list(filter(os.path.isdir, [f"models/{x}" for x in model_tests])))
d1.remove("models")
d = d2[1:3] + d1[:1]
print(d)
