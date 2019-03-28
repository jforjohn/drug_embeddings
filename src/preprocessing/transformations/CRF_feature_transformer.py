from sklearn.base import BaseEstimator, TransformerMixin
from models.rules.token_classifier import classify_token
from nltk import pos_tag

class CRFfeatureTransformer(BaseEstimator, TransformerMixin):
    def _extract_token_features(self, token, pos_tag, prefix=''):
        prefix = prefix + '_' if prefix else ''

        res = {
            f'{prefix}form': token['text'],
            f'{prefix}form_lower': token['text'].lower(),

            f'{prefix}suf3': token['text'][-3:],
            f'{prefix}suf4': token['text'][-4:],

            f'{prefix}is_upper': token['text'].isupper(),
            f'{prefix}is_title': token['text'].istitle(),
            f'{prefix}is_digit': token['text'].isdigit(),

            f'{prefix}pos_tag': pos_tag,
        }

        rule_class = classify_token(token)
        if rule_class:
            res[f'{prefix}rule_classification'] = rule_class.type

        return res

    def fit_transform(self, tokens):
        result = []

        pos_tags = [p for _, p in pos_tag([t['text'] for t in tokens])]

        for k in range(0, len(tokens)):
            feature = self._extract_token_features(tokens[k], pos_tags[k])

            if k > 0:
                feature.update(
                    self._extract_token_features(
                        tokens[k - 1], pos_tags[k - 1],
                        'previous'
                    )
                )
            else:
                feature['BoS'] = True

            if k < len(tokens)-1:
                feature.update(
                    self._extract_token_features(
                        tokens[k + 1], pos_tags[k + 1],
                        'next'
                    )
                )
            else:
                feature['EoS'] = True

            result.append(feature)

        return result
