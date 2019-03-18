from sklearn.base import BaseEstimator, TransformerMixin


class CRFfeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def _extract_token_features(self, token, prefix=''):
        prefix = prefix + '_' if prefix else ''
        return {
            f'{prefix}form': token,
            f'{prefix}form_lower': token.lower(),
            f'{prefix}suf3': token[-3:],
            f'{prefix}suf4': token[-4:],
            f'{prefix}is_upper': token.isupper(),
            f'{prefix}is_title': token.istitle(),
            f'{prefix}is_digit': token.isdigit()
        }

    def fit_transform(self, tokens):
        result = []

        for k in range(0, len(tokens)):
            feature = self._extract_token_features(tokens[k]['text'])

            if k > 0:
                feature.update(self._extract_token_features(tokens[k - 1]['text'], 'previous'))
            else:
                feature["BoS"] = True

            if k < len(tokens)-1:
                feature.update(self._extract_token_features(tokens[k + 1]['text'], 'next'))
            else:
                feature["EoS"] = True

            result.append(feature)

        return result
