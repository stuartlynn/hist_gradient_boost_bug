# Repo to reproduce an error when seralizing a HistGradientBoosting sklean model

This repo is an example of an issue with SKlearns HistGradientBoosting models. 

When the model is trained on a 64 bit environment and pickled, it fails to properly load in a 32 bit environment 
like pyodide 

See this ticket for more info : 

## Running the example

Generate the pickle file with 

```bash
python generate_pickle.py
```


Load the file in to pyodide and try to use it using 
```
npm install 
node index.js 
```

There is an example of a hacky fix for the problem which you can also run with as follow 

```
node index_with_hack.js
```
