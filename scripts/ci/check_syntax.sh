#!/bin/bash

# Exit if any command fails
set -e

# Trap the EXIT signal to print the last executed command when the script exits
trap 'echo "Error on line $LINENO: $BASH_COMMAND" >&2' ERR


# Find all staged c++ files
STAGED_FILES=$(git diff --cached --name-only --diff-filter=d | grep -E '\.h$' || true)

for FILE in $STAGED_FILES; do
  clang++ -I./narf/narf/include/ -fsyntax-only -Wno-deprecated $FILE 
  if [ $? -ne 0 ]; then
    echo "Syntax error in $FILE. Commit aborted."
    exit 1
  fi
done


# Find all staged YAML files
STAGED_FILES=$(git diff --cached --name-only --diff-filter=d | grep -E '\.ya?ml$' || true)

# Check syntax of each YAML file
for FILE in $STAGED_FILES; do
  python -c "import yaml, sys; yaml.safe_load(open('$FILE'))"
  if [ $? -ne 0 ]; then
    echo "Syntax error in $FILE. Commit aborted."
    exit 1
  fi
done


# Find all staged Python files
STAGED_FILES=$(git diff --cached --name-only --diff-filter=d | grep -E '\.py$' || true)

# Check for syntax errors in each staged Python file
for FILE in $STAGED_FILES; do
  # Use python's compile command to check for syntax errors
  python -m py_compile "$FILE"

  # Check the exit status of the previous command
  if [ $? -ne 0 ]; then
    echo "Syntax error in $FILE. Commit aborted."
    exit 1
  fi
done

echo "All syntax checks passed"
exit 0


