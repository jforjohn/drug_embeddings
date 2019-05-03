import json

def load_config_file(nfile, abspath=False):
    """
    Read the configuration from a json file

    :param abspath:
    :param nfile:
    :return: json
    """
    ext = '.json' if 'json' not in nfile else ''
    pre = '' if abspath else './'
    fp = open(pre + nfile + ext, 'r')

    s = ''

    for l in fp:
        s += l

    return json.loads(s)
