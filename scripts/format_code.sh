#!/bin/bash -e
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Run this script at project root by "./dev/linter.sh" before you commit

#vergte() {
#  [ "$2" = "$(echo -e "$1\\n$2" | sort -V | head -n1)" ]
#}

{
  black --version | grep -E "21.5b0" > /dev/null
} || {
  echo "Linter requires 'black==21.5b0' !"
  exit 1
}

set -v


echo "Running black ..."
# black -l 100 .
black -l 120 .

docformatter -ir .



command -v arc > /dev/null && arc lint
