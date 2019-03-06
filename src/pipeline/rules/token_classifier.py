from models import DrugEntity

drugs = ['ine', 'ide', 'cin', 'ole', 'one', 'ate', 'rin', 'ium', 'oin',
         'xin', 'vir', 'tin', 'cid', 'lam', 'lol', 'fen', 'hol', 'nol', 'lin', 'pin']
groups = ['nts', 'ics', 'nes', 'ids', 'ors', 'ugs', 'ves', 'ers', 'tes',
          'des', 'ant', 'sts', 'ins', 'tic', 'ens', 'tor', 'lis', 'ons', 'oid', 'n d']


def classify_token(token):
    if token['text'].isupper():
        return DrugEntity(
            offsets=[token['char_offset']],
            type='brand',
            text=token['text']
        )
    elif any(token['text'].endswith(s) for s in groups):
        return DrugEntity(
            offsets=[token['char_offset']],
            type='group',
            text=token['text']
        )
    elif any(token['text'].endswith(s) for s in drugs):
        return DrugEntity(
            offsets=[token['char_offset']],
            type='drug',
            text=token['text']
        )
    return None

def classify_tokens(tokens):
    res = []
    prev_token = None
    for drug_entity in [classify_token(t) for t in tokens]:
        if drug_entity is not None:
            if prev_token is None:
                prev_token = drug_entity
            else:
                if prev_token.type == drug_entity.type:
                    prev_token.text += f' {drug_entity.text}'
                    prev_token.offsets[0][1] = drug_entity.offsets[0][1]
                else:
                    res.append(prev_token)
                    prev_token = drug_entity
        elif prev_token is not None:
            res.append(prev_token)
            prev_token = None
    return res
