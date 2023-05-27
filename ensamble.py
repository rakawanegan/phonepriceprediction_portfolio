import pandas as pd
import glob

csv_files = glob.glob('results/submit/*.csv')
csv_files = ["results/submit/knnsamplepredict.csv", "results/submit/lgbmsamplepredict.csv"]

dfs = []
for filename in csv_files:
    dfs.append(pd.read_csv(filename, header=None, names=["idx",'predict']))
df = pd.concat(dfs,axis=1)["predict"]

submit = pd.DataFrame(df.mean(axis=1), columns=['predict'])
submit['predict']=submit['predict'].map(lambda x: int(max(min(x, 3), 0)))
submit.index = dfs[0]["idx"]


submit.to_csv(f"results/submit/ensamblesubmit.csv",
              header=False)