import pandas as pd
s1 = pd.Series(['a', 'b'], ['c', 'd'])
s2 = pd.Series(['e', 'f'], ['g', 'h'])
print(pd.concat([s1, s2], axis = 1))