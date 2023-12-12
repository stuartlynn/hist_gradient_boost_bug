from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.datasets import load_diabetes
import sys
import sklearn
import pickle 

print("python version", sys.version)
print("sklearn version",sklearn.__version__ )

X, y = load_diabetes(return_X_y=True)
est = HistGradientBoostingRegressor().fit(X, y)
with open("hgb.pickle","wb") as f :
    pickle.dump(est,f)

with open("hgb.pickle","rb") as f:
    reloaded = pickle.load(f)

result = reloaded.predict(X)
print("Reloaded result ",result)
