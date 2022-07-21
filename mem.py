import os
import time
os.system("rm -rf mem.txt")
while True:
    os.system("ps aux --sort pmem >> mem.txt")
    os.system('echo "----------" >> mem.txt')
    os.system("ls -l reports/tests_torch/ >> mem.txt")
    os.system('echo "----------" >> mem.txt')
    os.system("date >> mem.txt")
    os.system('echo "================" >> mem.txt')
    time.sleep(5)