#!/bin/bash
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

TNAME=$(basename "$0" .sh)

for NP in 1 2; do
  COUNT=$(grep -c "iterations=" "${TNAME}.out.${NP}")
  if [ "$COUNT" != "7" ]; then
    echo "Incorrect number of runs in ${TNAME}.out.${NP}" >&2
  fi
done
