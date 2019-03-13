from sklearn.base import BaseEstimator, TransformerMixin

class CRFfeatureTransformer(BaseEstimator, TransformerMixin):
   def __init__(self):
      pass

   def fit_transform(self, tokens):
      result = []
      # for each token, generate list of features and add it to the result
      for k in range(0, len(tokens)):

         t = tokens[k]['text']

         feature = {}
         feature["form"] = t
         feature["formlower"] = t.lower()
         feature["suf3"] = t[-3:]
         feature["suf4"] = t[-4:]
         feature["isUpper"] = t.isupper()
         feature["isTitle"] = t.istitle()
         feature["isDigit"] = t.isdigit()

         if k>0 :
            tPrev = tokens[k-1]['text']
            feature["formPrev"] = tPrev
            feature["formlowerPrev"] = tPrev.lower()
            feature["suf3Prev"] = tPrev[-3:]
            feature["suf4Prev"] = tPrev[-4:]
            feature["isUpperPrev"] = tPrev.isupper()
            feature["isTitlePrev"] = tPrev.istitle()
            feature["isDigitPrev"] = tPrev.isdigit()
         else :
            feature["BoS"] = True

         if k<len(tokens)-1 :
            tNext = tokens[k+1]['text']
            feature["formNext"] = tNext
            feature["formlowerNext"] = tNext.lower()
            feature["suf3Next"] = tNext[-3:]
            feature["suf4Next"] = tNext[-4:]
            feature["isUpperNext"] = tNext.isupper()
            feature["isTitleNext"] = tNext.istitle()
            feature["isDigitNext"] = tNext.isdigit()
         else:
            feature["EoS"] = True


         result.append(feature)
      
      return result