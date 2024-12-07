## Crystal ballin 

### Setup

using UV to manage python depedencies:

To initialize a virtual Env:

```
➜  crystal-ballin-backend git:(main) ✗ uv venv

Using CPython 3.9.20
Creating virtual environment at: .venv
Activate with: source .venv/bin/activate
```

Since this deploys to heroku, we need to maintain pip depedencies:

for all installation of future files, must use:

`uv pip install flask flask-cors prophet pandas yfinance python-dotenv`

freeze deps:

`uv pip freeze > requirements.txt`

to run:

```
# For Unix/MacOS:
.venv/bin/python app.py

# For Windows:
.venv\Scripts\python app.py
```

