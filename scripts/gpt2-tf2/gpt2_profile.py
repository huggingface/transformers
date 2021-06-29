import sys
import pandas as pd
profile_dir = sys.argv[1]
df = pd.read_csv(profile_dir+'results.stats.csv')
print('Total time for one step GPT2', sum(df["TotalDurationNs"])*1e-9, 's')
