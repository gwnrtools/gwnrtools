---
title: Development
layout: default
nav_order: 7
permalink: /docs/development/
---

# Development
{: .no_toc }

1. TOC
{:toc}

## Repository layout

```
gwnr/
├── gwnr/               # The Python package (import gwnr)
│   ├── analysis/       #   matches, fitting factors, PSDs, catalogs, template banks
│   ├── cosmo/          #   cosmological conversions
│   ├── data/           #   bundled detector noise curves
│   ├── graph/          #   plotting
│   ├── nr/             #   numerical relativity (SXS, SpEC, SpECTRE)
│   ├── stats/          #   inference configs, samplers, Fisher matrices
│   ├── utils/          #   general helpers, type conversions
│   ├── waveform/       #   generation, alignment, hybridization, eccentricity
│   └── workflow/       #   HTCondor / batch-analysis generation
├── bin/                # Command-line tools (subset installed via setup.py)
├── notebooks/          # Jupyter notebooks (tutorials + research)
├── tutorials/          # HTML renderings of tutorial notebooks
├── docs/               # This documentation site (Markdown, Jekyll)
├── setup.py            # Package metadata, installed scripts
└── requirements.txt    # Dependencies
```

## Contributing

1. Fork and clone the repository, then install in editable mode:
   ```bash
   pip install -e .
   ```
2. Code style is [black](https://github.com/psf/black); please format touched files before
   committing.
3. Open pull requests against `master` at
   [gwnr/gwnr](https://github.com/gwnr/gwnr).

## Running the tests

The unit-test suite lives in `tests/` and uses `pytest` (configuration in `setup.cfg`):

```bash
python -m pytest tests/            # full suite
python -m pytest tests/ -m "not slow"   # skip waveform-generation/sampler tests
```

Tests marked `slow` generate waveforms or run MCMC samplers. The suite requires the GW software
stack (LALSuite, PyCBC, `igwn-ligolw`, `lscsoft-glue`); tests for optional functionality skip
automatically when their dependency is missing.

Continuous integration runs the full suite on every pull request and on pushes to `master` via
GitHub Actions
([`.github/workflows/tests.yml`](https://github.com/gwnr/gwnr/blob/master/.github/workflows/tests.yml)),
on Python 3.10 and 3.11.

## This documentation site

The site is plain Markdown rendered by GitHub Pages with Jekyll and the
[just-the-docs](https://just-the-docs.github.io/just-the-docs/) remote theme — no build step is
required in the repository. Configuration lives in
[`_config.yml`](https://github.com/gwnr/gwnr/blob/master/_config.yml); pages are the
root `index.md` plus everything under `docs/`.

### Publishing on GitHub Pages

In the repository settings on GitHub: **Settings → Pages → Build and deployment**, choose
*Deploy from a branch*, branch `master`, folder `/ (root)`. The site is then served at
`https://<org>.github.io/gwnr/`. Any push to `master` republishes automatically.

### Previewing locally

```bash
gem install bundler jekyll
cat > Gemfile <<'EOF'
source "https://rubygems.org"
gem "github-pages", group: :jekyll_plugins
EOF
bundle install
bundle exec jekyll serve
# open http://localhost:4000/gwnr/
```

### Updating the API reference

The API pages under `docs/api/` are written by hand from the package sources. When adding or
changing public functions/classes, please update the corresponding page.

## License

GNU General Public License — see
[LICENSE](https://github.com/gwnr/gwnr/blob/master/LICENSE).
