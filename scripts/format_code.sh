#!/bin/bash -e
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Run this script at project root by "./dev/linter.sh" before you commit

#vergte() {
#  [ "$2" = "$(echo -e "$1\\n$2" | sort -V | head -n1)" ]
#}

BLACK_VERSION="21.11b1"
{
  black --version | grep -E "$BLACK_VERSION" > /dev/null
} || {
  echo "Linter requires 'black==$BLACK_VERSION' !"
  exit 1
}

set -v


echo "Running black ..."
# black -l 100 .
black -l 120 .

docformatter -ir .



command -v arc > /dev/null && arc lint
