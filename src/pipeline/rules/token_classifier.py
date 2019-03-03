from models import Entity

drugs = ['ine', 'ide', 'cin', 'ole', 'one', 'ate', 'rin', 'ium', 'oin',
         'xin', 'vir', 'tin', 'cid', 'lam', 'lol', 'fen', 'hol', 'nol', 'lin', 'pin']
groups = ['nts', 'ics', 'nes', 'ids', 'ors', 'ugs', 'ves', 'ers', 'tes',
          'des', 'ant', 'sts', 'ins', 'tic', 'ens', 'tor', 'lis', 'ons', 'oid', 'n d']


def classify_token(token):
    if token['text'].isupper():
        return Entity(
            char_offset=token['char_offset'],
            type='brand',
            text=token['text']
        )
    elif any(token['text'].endswith(s) for s in groups):
        return Entity(
            char_offset=token['char_offset'],
            type='group',
            text=token['text']
        )
    elif any(token['text'].endswith(s) for s in drugs):
        return Entity(
            char_offset=token['char_offset'],
            type='drug',
            text=token['text']
        )
    return None
