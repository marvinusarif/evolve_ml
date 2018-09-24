import pandas as pd
d = {'x1': [1, 4, 4, 5, 6],
     'x2': [0, 0, 8, 2, 4],
     'x3': [2, 8, 8, 10, 12],
     'x4': [-1, -4, -4, -4, -5]}
df = pd.DataFrame(data = d)
print("Data Frame")
print(df)
print()

print("Correlation Matrix")
print(df.corr())
print()


def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
print(get_top_abs_correlations(df, 3))