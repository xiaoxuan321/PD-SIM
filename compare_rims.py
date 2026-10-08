import pandas as pd

rims = pd.read_csv(r'E:\RIMS-main\Productions\results\rims_simple_index_det\tst_Productions.csv')
simod = pd.read_csv(r'E:\Simod-5.1.60\resources\event_logs\Production\Production_test.csv')
dsim = pd.read_csv(r'D:\Download\5734443\reproducibility_package\reproducibility_package_caise_22\event_logs\test_partitions\tst_Production.csv')
dpsim = pd.read_csv(r'E:\DPSIM-1\DP-SIM123full\output_files\20260723_4D9C16B6_4F60_40DC_AB63_F267F7F1B2DD\tst_Production_dpsim.csv')

print('=== RIMS columns ===')
print(list(rims.columns))
print(f'Rows: {len(rims)}, Cases: {rims["caseid"].nunique() if "caseid" in rims.columns else "N/A"}')
print(f'Sample:')
print(rims.head(3).to_string())
print()

# Detect column mapping
col_map = {}
for c in rims.columns:
    cl = c.lower()
    if cl in ('caseid', 'case:id', 'case_id'):
        col_map['caseid'] = c
    elif cl in ('activity', 'task', 'concept:name'):
        col_map['task'] = c
    elif cl in ('start_time', 'start_timestamp', 'start'):
        col_map['start'] = c
    elif cl in ('end_time', 'end_timestamp', 'end', 'time:timestamp'):
        col_map['end'] = c
    elif cl in ('user', 'org:resource', 'resource'):
        col_map['user'] = c

print(f'Detected column mapping: {col_map}')
print()

# Rename for comparison
rims_r = rims.rename(columns={v: k for k, v in col_map.items()})
simod_r = simod.rename(columns={'activity': 'task', 'start_time': 'start', 'end_time': 'end', 'org:resource': 'user'})
dsim_r = dsim.rename(columns={'start_timestamp': 'start', 'end_timestamp': 'end'})
dpsim_r = dpsim.rename(columns={'start_timestamp': 'start', 'end_timestamp': 'end'})

# Basic info
print('=== Basic info ===')
for name, df in [('RIMS', rims_r), ('Simod', simod_r), ('D-SIM', dsim_r), ('DP-SIM', dpsim_r)]:
    ci = df['caseid'].nunique() if 'caseid' in df.columns else '?'
    print(f'{name}: {len(df)} rows, {ci} cases')
print()

# Duplicate check
print('=== Duplicate check ===')
for name, df in [('RIMS', rims_r), ('Simod', simod_r), ('D-SIM', dsim_r), ('DP-SIM', dpsim_r)]:
    check_cols = [c for c in ['caseid', 'task', 'start', 'end', 'user'] if c in df.columns]
    dups = df[df.duplicated(subset=check_cols, keep=False)]
    print(f'{name}: {len(dups)} duplicate rows')
print()

# Timestamp range
print('=== Timestamp range ===')
for name, df in [('RIMS', rims_r), ('Simod', simod_r), ('D-SIM', dsim_r), ('DP-SIM', dpsim_r)]:
    if 'start' in df.columns and 'end' in df.columns:
        print(f'{name}: start [{df["start"].min()} ~ {df["start"].max()}]  end [{df["end"].min()} ~ {df["end"].max()}]')
print()

# CaseID comparison
sets = {}
for name, df in [('RIMS', rims_r), ('Simod', simod_r), ('D-SIM', dsim_r), ('DP-SIM', dpsim_r)]:
    sets[name] = set(df['caseid'].unique())

print('=== CaseID overlap ===')
all_common = sets['RIMS'] & sets['Simod'] & sets['D-SIM'] & sets['DP-SIM']
print(f'All 4 common: {len(all_common)}')
for n1, n2 in [('RIMS','Simod'), ('RIMS','D-SIM'), ('RIMS','DP-SIM')]:
    common = sets[n1] & sets[n2]
    only1 = sets[n1] - sets[n2]
    only2 = sets[n2] - sets[n1]
    print(f'{n1} vs {n2}: common={len(common)}, only {n1}={len(only1)}, only {n2}={len(only2)}')
print()

# Tuple comparison
def to_tuples(df, cols):
    s = set()
    for _, r in df.iterrows():
        s.add(tuple(str(r[x]) for x in cols))
    return s

cols = ['caseid', 'task', 'user', 'start', 'end']
available_cols = [c for c in cols if c in rims_r.columns and c in simod_r.columns and c in dsim_r.columns and c in dpsim_r.columns]
print(f'=== Tuple comparison (cols={available_cols}) ===')

tuples = {}
for name, df in [('RIMS', rims_r), ('Simod', simod_r), ('D-SIM', dsim_r), ('DP-SIM', dpsim_r)]:
    tuples[name] = to_tuples(df, available_cols)
    print(f'{name}: {len(tuples[name])} unique tuples')
print()

for n1, n2 in [('RIMS','Simod'), ('RIMS','D-SIM'), ('RIMS','DP-SIM'), ('Simod','D-SIM'), ('D-SIM','DP-SIM')]:
    common = tuples[n1] & tuples[n2]
    only1 = tuples[n1] - tuples[n2]
    only2 = tuples[n2] - tuples[n1]
    print(f'{n1} vs {n2}: common={len(common)}, only {n1}={len(only1)}, only {n2}={len(only2)}')
    if only1 and len(only1) <= 5:
        for t in sorted(list(only1)):
            print(f'  Only {n1}: {t}')
    if only2 and len(only2) <= 5:
        for t in sorted(list(only2)):
            print(f'  Only {n2}: {t}')
