class Writer(object):
    def __init__(self, path=None):
        self.path = path

    def call(self, df, col_names=['drugs', 'pairs']):
        cols = set(df.columns) & set([*col_names, 'id'])
        res = ''

        for _, row in df[cols].iterrows():
            item_names = set(row.index) - {'id'}
            for item_name in item_names:
                for item in row[item_name]:
                    res += f'{row.id}|{item.to_output()}\n'

        if self.path is not None:
            with open(self.path, 'w') as f:
                f.writelines(res)
        return res
