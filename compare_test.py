import pandas as pd

f_caise = r'D:\Download\5734443\reproducibility_package\reproducibility_package_caise_22\event_logs\test_partitions\tst_PurchasingExample.csv'
f_dpsim = r'E:\DPSIM-1\DP-SIM123full\output_files\20260724_361DA90D_7D3E_4CEB_BE35_7A928CAA865D\tst_PurchasingExample_dpsim.csv'

df1 = pd.read_csv(f_caise)
df2 = pd.read_csv(f_dpsim)

print('=' * 60)
print('1. 基本信息')
print('=' * 60)
print(f'CAISE 22: {df1.shape[0]} rows, {df1.shape[1]} cols -> {list(df1.columns)}')
print(f'DP-SIM:   {df2.shape[0]} rows, {df2.shape[1]} cols -> {list(df2.columns)}')

# Common columns
common_cols = ['caseid', 'task', 'user', 'end_timestamp', 'start_timestamp']
extra_cols = sorted(set(df1.columns) - set(df2.columns))
print(f'\n共同列: {common_cols}')
print(f'CAISE 22 额外列: {extra_cols}')

print('\n' + '=' * 60)
print('2. CaseID 对比')
print('=' * 60)
cases1 = set(df1['caseid'].unique())
cases2 = set(df2['caseid'].unique())
print(f'CAISE 22 唯一 case: {len(cases1)}')
print(f'DP-SIM   唯一 case: {len(cases2)}')
print(f'共同 case: {len(cases1 & cases2)}')
only1 = sorted(cases1 - cases2)
only2 = sorted(cases2 - cases1)
print(f'仅 CAISE 22: {only1 if only1 else "无"}')
print(f'仅 DP-SIM:   {only2 if only2 else "无"}')

print('\n' + '=' * 60)
print('3. 行排序对比')
print('=' * 60)
print(f'CAISE 22 前5行 caseid: {list(df1["caseid"].head())}')
print(f'DP-SIM   前5行 caseid: {list(df2["caseid"].head())}')
print(f'CAISE 22 是否按caseid分组: {all(df1.groupby("caseid").apply(lambda g: list(g.index) == list(range(g.index.min(), g.index.max()+1))))}')
print(f'DP-SIM   end_timestamp是否递增: {(df2["end_timestamp"].values[1:] >= df2["end_timestamp"].values[:-1]).all()}')

print('\n' + '=' * 60)
print('4. 共同5列数据对比（排序后逐行比较）')
print('=' * 60)

sort_cols = ['caseid', 'start_timestamp', 'end_timestamp', 'task', 'user']
df1_s = df1[common_cols].sort_values(sort_cols).reset_index(drop=True)
df2_s = df2[common_cols].sort_values(sort_cols).reset_index(drop=True)

print(f'CAISE 22 行数: {len(df1_s)}')
print(f'DP-SIM   行数: {len(df2_s)}')

if len(df1_s) == len(df2_s):
    diff_mask = (df1_s != df2_s)
    n_diffs = diff_mask.any(axis=1).sum()
    print(f'不同行数: {n_diffs} / {len(df1_s)}')
    if n_diffs > 0:
        diff_idx = diff_mask.any(axis=1)
        print('\n差异详情（前20条）:')
        for i, idx in enumerate(df1_s[diff_idx].index[:20]):
            print(f'\n  [{i+1}] 行索引 {idx}:')
            for col in common_cols:
                v1, v2 = df1_s.loc[idx, col], df2_s.loc[idx, col]
                if str(v1) != str(v2):
                    print(f'    {col}: CAISE=[{v1}] vs DP-SIM=[{v2}]')
    else:
        print('\n*** 完全一致！所有行在5个共同列上完全匹配 ***')
else:
    print(f'行数不同，无法直接逐行比较')
    # Per-case comparison
    print('\n按case逐案例对比:')
    diff_cases = 0
    for cid in sorted(cases1 & cases2):
        g1 = df1[df1['caseid'] == cid][common_cols].sort_values(sort_cols).reset_index(drop=True)
        g2 = df2[df2['caseid'] == cid][common_cols].sort_values(sort_cols).reset_index(drop=True)
        if len(g1) != len(g2):
            diff_cases += 1
            if diff_cases <= 10:
                print(f'  Case {cid}: CAISE={len(g1)}行, DP-SIM={len(g2)}行')
        elif not (g1.values == g2.values).all():
            diff_cases += 1
            if diff_cases <= 10:
                print(f'  Case {cid}: 行数相同({len(g1)})但数据不同')
                for j in range(len(g1)):
                    if not (g1.iloc[j] == g2.iloc[j]).all():
                        for col in common_cols:
                            if str(g1.iloc[j][col]) != str(g2.iloc[j][col]):
                                print(f'    行{j} {col}: CAISE=[{g1.iloc[j][col]}] vs DP-SIM=[{g2.iloc[j][col]}]')
    total_common = len(cases1 & cases2)
    print(f'  ... 共 {total_common} 个共同 case 中有 {diff_cases} 个存在差异')

print('\n' + '=' * 60)
print('5. 总结')
print('=' * 60)
same_rows = len(df1) == len(df2)
same_cases = cases1 == cases2
if same_rows and same_cases and len(df1_s) == len(df2_s) and not (df1_s != df2_s).any(axis=1).any():
    print('两个文件在核心5列上完全一致（忽略额外列和排序差异）')
else:
    print('两个文件不完全一致')
    if not same_rows:
        print(f'  - 行数不同: CAISE={len(df1)}, DP-SIM={len(df2)}')
    if not same_cases:
        print(f'  - case集合不同: {len(only1)}个仅在CAISE, {len(only2)}个仅在DP-SIM')
