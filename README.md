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

# Usage

