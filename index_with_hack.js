const {loadPyodide}= require("pyodide")
const fs = require('fs')


let program =`
  import pickle 
  import sys
  import sklearn
  import numpy as np
  from sklearn.ensemble import HistGradientBoostingRegressor
  from sklearn.datasets import load_diabetes

  Y_DTYPE = np.float64
  X_DTYPE = np.float64
  X_BINNED_DTYPE = np.uint8  # hence max_bins == 256
  # dtype for gradients and hessians arrays
  G_H_DTYPE = np.float32
  X_BITSET_INNER_DTYPE = np.uint32

  PREDICTOR_RECORD_DTYPE_2 = np.dtype([
      ('value', Y_DTYPE),
      ('count', np.uint32),
      ('feature_idx', np.int32),
      ('num_threshold', X_DTYPE),
      ('missing_go_to_left', np.uint8),
      ('left', np.uint32),
      ('right', np.uint32),
      ('gain', Y_DTYPE),
      ('depth', np.uint32),
      ('is_leaf', np.uint8),
      ('bin_threshold', X_BINNED_DTYPE),
      ('is_categorical', np.uint8),
      # The index of the corresponding bitsets in the Predictor's bitset arrays.
      # Only used if is_categorical is True
      ('bitset_idx', np.uint32)
  ])



  print("python version", sys.version)
  print("sklearn version",sklearn.__version__ )

  X, y = load_diabetes(return_X_y=True)
  with open("/hgb.pickle","rb") as f:
    model = pickle.load(f)

  for i,_ in enumerate(model._predictors):
    model._predictors[i][0].nodes = model._predictors[i][0].nodes.astype(PREDICTOR_RECORD_DTYPE_2)

  prediction = model.predict(X)
  print(prediction)
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

