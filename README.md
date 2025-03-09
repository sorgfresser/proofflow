# Setup

I used poetry for dependency management. To install the deps, run 
```bash
poetry install
```

Obviously, this also requires lean. Use elan for this ideally.

For now, you need to downgrade the repl manually because the LeanDojo benchmark uses an old Lean version. This involves going to your venv and resetting the commit as follows:
```bash
IFS=. read -r var1 var2 var3 <<< $(python3 --version)
cd $(dirname $(which python3))/../lib/python3.$var2/site-packages/repl && git checkout 4fc1e6d1dda170e8f0a6b698dd5f7e17a9cf52b4 && lake build
```

To add a dependency you introduced, run
```bash
poetry add <dependencyname>
```

You also need to get the LeanDojo data. For this, obtain it from [Zenodo](https://zenodo.org/records/12740403). Unzip the file in the root folder of this repo, such that `leandojo_benchmark_4` is on the same level as `leanproject` and `proofflow`. 

You might want to run torch with cuda binaries. In that case, simply replace the line on torch with a one without source
```bash
sed -i '/^torch = { version = "2\.6\.0"/ s/.*/torch = "^2.6.0"/' pyproject.toml && rm poetry.lock && poetry lock
```

And finally, install mamba using the following command:
```bash
poetry run pip3 install setuptools wheel packaging
poetry run pip3 install --no-use-pep517 causal-conv1d
poetry run pip3 install mamba-ssm
```

# Usage

