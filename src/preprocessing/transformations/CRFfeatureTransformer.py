from sklearn.base import BaseEstimator, TransformerMixin

class CRFfeatureTransformer(BaseEstimator, TransformerMixin):
   def __init__(self):
      pass

   def fit_transform(self, tokens):
      result = []
      # for each token, generate list of features and add it to the result
      for k in range(0, len(tokens)):
         tokenFeatures = []
         t = tokens[k]['text']
         tokenFeatures.extend([tokens[k]['char_offset'][0],
                              tokens[k]['char_offset'][1]])

         tokenFeatures.append("form="+t)
         tokenFeatures.append("formlower="+t.lower())
         tokenFeatures.append("suf3="+t[-3:])
         tokenFeatures.append("suf4="+t[-4:])
         if (t.isupper()) : tokenFeatures.append("isUpper")
         if (t.istitle()) : tokenFeatures.append("isTitle")
         if (t.isdigit()) : tokenFeatures.append("isDigit")

         if k>0 :
            tPrev = tokens[k-1]['text']
            tokenFeatures.append("formPrev="+tPrev)
            tokenFeatures.append("formlowerPrev="+tPrev.lower())
            tokenFeatures.append("suf3Prev="+tPrev[-3:])
            tokenFeatures.append("suf4Prev="+tPrev[-4:])
            if (t.isupper()) : tokenFeatures.append("isUpperPrev")
            if (t.istitle()) : tokenFeatures.append("isTitlePrev")
            if (t.isdigit()) : tokenFeatures.append("isDigitPrev")
         else :
            tokenFeatures.append("BoS")

         if k<len(tokens)-1 :
            tNext = tokens[k+1]['text']
            tokenFeatures.append("formNext="+tNext)
            tokenFeatures.append("formlowerNext="+tNext.lower())
            tokenFeatures.append("suf3Next="+tNext[-3:])
            tokenFeatures.append("suf4Next="+tNext[-4:])
            if (t.isupper()) : tokenFeatures.append("isUpperNext")
            if (t.istitle()) : tokenFeatures.append("isTitleNext")
            if (t.isdigit()) : tokenFeatures.append("isDigitNext")
         else:
            tokenFeatures.append("EoS")
      
         result.append(tokenFeatures)
      
      return result