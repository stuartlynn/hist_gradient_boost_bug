const {loadPyodide}= require("pyodide")
const fs = require('fs')


let program =`
  import pickle 
  import sys
  import sklearn
  from sklearn.ensemble import HistGradientBoostingRegressor
  from sklearn.datasets import load_diabetes

  print("python version", sys.version)
  print("sklearn version",sklearn.__version__ )
  X, y = load_diabetes(return_X_y=True)
  with open("/hgb.pickle","rb") as f:
    model = pickle.load(f)

  model.predict(X)
`

async function run_example(){
  
  let pyodide = await loadPyodide();

  await pyodide.loadPackage(["micropip"])  
  const micropip = pyodide.pyimport("micropip")
  await micropip.install("scikit-learn")
  
  const model = fs.readFileSync("./hgb.pickle")
  pyodide.FS.writeFile("/hgb.pickle", model);

  let results = await pyodide.runPythonAsync(program)
}

run_example().then(()=> process.exit(0))
