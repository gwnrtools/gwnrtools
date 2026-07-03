---
title: gwnr.workflow
layout: default
parent: API Reference
nav_order: 9
permalink: /docs/api/workflow/
---

# `gwnr.workflow` — workflow automation
{: .no_toc }

Programmatic construction of batch analyses and HTCondor workflows; the engine behind the
`gwnr_create_*_workflow` [command-line tools]({{ site.baseurl }}/docs/cli/). Source:
[`gwnr/workflow/`](https://github.com/gwnrtools/gwnrtools/tree/master/gwnr/workflow).

1. TOC
{:toc}

## `utils.py` — configuration parsing

**`class WorkflowConfigParserBase(pycbc.workflow.configuration.InterpolatingConfigParser)`** —
base INI parser for workflow configuration files; `get_ini_opts(confs, section)` extracts
command-line options from a config section.

## `condor.py` — HTCondor job/node classes

Thin wrappers over `glue.pipeline` Condor classes:

- **`class BaseJob(CondorDAGJob, CondorJob)`** — common job setup (universe, executable, logs)
- **`class BanksimNode(CondorDAGNode)`**, **`class BanksimCombineNode(CondorDAGNode)`** — nodes
  for bank-simulation jobs and their match-combination step
- **`class FaithsimNode(CondorDAGNode)`** — faithfulness-simulation jobs
- **`class InferenceJob(CondorDAGJob, CondorJob)`** — parameter-estimation jobs

## `inference.py` — batch inference framework

- **`class OneInferenceAnalysis`** — one parameter-estimation run: resolves analysis
  directories, executables (injection generator, inference engine, plotting), log dirs; `setup()`
  creates the run directory and `write_run_script(...)` emits the shell script that executes the
  analysis.
- **`class BatchInferenceAnalyses`** — a campaign of such runs: `setup_runs()` creates one
  `OneInferenceAnalysis` per injection/event, with `name_run_dir(...)` defining the layout and
  `get_run_tag()` labeling the campaign.

## `pycbc_inference.py` — PyCBC Inference backends

Concrete subclasses for `pycbc_inference`:

- **`PycbcInferenceInjectionAnalysis`** / **`PycbcInferenceOnInjectionBatch`** — runs on
  synthetic injections.
- **`PycbcInferenceEventAnalysis`** / **`PycbcInferenceOnEventBatch`** — runs on real (public)
  events; `fetch_all_data()` and `fetch_all_psds()` download the strain and PSDs around each
  event via the
  [`gwnr.analysis` catalog classes]({{ site.baseurl }}/docs/api/analysis/#gw_transient_catalogpy--gw-event-catalogs).

## `bilby.py` — Bilby backends

- **`class BilbyInferenceConfigParser(WorkflowConfigParserBase)`** — reads the campaign INI:
  inference options, interferometer list, source type, prior lines, injection parameters; can
  emit `injection.ini` files.
- **`BilbyInferenceInjectionAnalysis`** / **`BilbyInferenceEventAnalysis`** — write per-run Bilby
  prior files and run scripts.
- **`BilbyOnInjectionBatch`** / **`BilbyOnEventBatch`** — the corresponding campaign classes.
