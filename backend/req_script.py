import ast

# Replace with your script filename
SCRIPT_FILE = "company_scoring.py"
REQUIREMENTS_FILE = "requirements.txt"

# Parse the script
with open(SCRIPT_FILE, "r") as f:
    tree = ast.parse(f.read(), filename=SCRIPT_FILE)

imports = set()

for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        for n in node.names:
            imports.add(n.name.split('.')[0])
    elif isinstance(node, ast.ImportFrom):
        if node.module:
            imports.add(node.module.split('.')[0])

# Filter out standard library modules
# Optional: you can manually remove stdlib modules if you know them
# e.g., stdlibs = {"os", "sys", "math", ...}
# imports = imports - stdlibs

# Write to requirements.txt
with open(REQUIREMENTS_FILE, "w") as f:
    for lib in sorted(imports):
        f.write(lib + "\n")

print(f"Requirements saved to {REQUIREMENTS_FILE}")
