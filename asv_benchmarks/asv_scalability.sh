#!/usr/bin/env bash
asv run -b ArgKmin -e pairwise-distances-argkmin^! | tee pda.txt
asv run -b ArgKmin -e 1.0^! | tee 1.0.txt

