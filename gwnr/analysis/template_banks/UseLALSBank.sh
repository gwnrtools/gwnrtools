#!/usr/bin/env bash

lalapps_cbc_sbank  --reference-psd  psd-T1200307v4_H1.xml 
--instrument H1 --fhigh-max 2048. --flow 30. --mass1-min 3.5 --mass1-max 3.8
--mass2-min 1.2 --mass2-max 1.4 --mtotal-min 3.0 --mtotal-max 10.0
--mratio-min 1.0 --spin1-max 0.9 --spin1-min 0.5 --spin2-max 0.4
--spin2-min 0. --approximant IMRPhenomPv2 --match-min 0.97
--cache-waveforms  --iterative-match-df-max 0.5 --coarse-match-df 8.0
--convergence-threshold 50 --seed 100052 --output-filename  TEST2.xml
--verbose
