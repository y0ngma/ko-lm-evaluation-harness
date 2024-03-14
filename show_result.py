import pandas as pd
import glob, os, json

# results/korean_origin_bench/20b/00_shot.json
def _get_metric_name(v):
    metrics = ['f1', 'macro_f1', 'acc_norm', 'acc']
    for m in metrics:
        if v.get(m):
            return {
                'metric': m,
                'value': v[m],
            }

def get_df_klue(path, model_name=''):
    data = []
    for i in ['0', '5', '10', '50']:
        shot = f'{path}/{i}_shot.json'
        try:
            data.append(
                {
                    f"{k} ({_get_metric_name(v)['metric']})": _get_metric_name(v)['value']
                    for k, v in json.load(open(shot))['results'].items()
                }
            )
        except FileNotFoundError:
            pass
    df = pd.DataFrame(data, index=[0, 5, 10, 50][:len(data)]).T
    print(df.to_markdown())
    return df

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
# get_df_klue('results/klue_etc_bench/home/jovyan/beomi/llama2-koen-13b/60b', 'llama2-koen-13b')
# get_df_klue('results/all/LDCC/LDCC-SOLAR-10.7B', 'LDCC-SOLAR-10.7B')
# get_df_klue('/data/yhjeong/home/output/eval/all/LDCC/LDCC-SOLAR-10.7B', 'LDCC-SOLAR-10.7B')
# get_df_klue('/data/yhjeong/home/output/eval/kobest_hellaswag', 'LDCC-SOLAR-10.7B')

# various_models = sorted(glob.glob(f'{PROJECT_DIR}/results/all/*/*'))
# various_models = sorted(glob.glob("/data/yhjeong/home/output/eval/kobest_hellaswag/*/*"))
# various_models = sorted(glob.glob("/data/yhjeong/home/output/eval/kobest_boolq/*/*"))

tasks = ["kobest_boolq", "kobest_hellaswag"]
various_models = list()
for task in tasks:
    various_models += sorted(glob.glob(f'/data/yhjeong/home/output/eval/{task}/*/*'))

for model in various_models:
    print(model)
    df = get_df_klue(model)
    print()