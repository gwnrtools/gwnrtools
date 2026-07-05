# gwnr

A collection of tools for academic research in gravitational-wave astronomy & astrophysics, and numerical relativity

**📖 Documentation: <https://gwnr.github.io/gwnr/>**

The Python package is named `gwnr` and provides:

- **Waveform tools** — generation, conditioning, alignment, hybridization of inspiral and
  merger–ringdown modes, eccentricity measurement, tidal corrections
- **Data analysis** — faithfulness and fitting-factor calculations, PSD handling, GW transient
  catalog access, stochastic template-bank construction
- **Numerical relativity** — SXS/SpEC waveform handling, SpEC & SpECTRE simulation output parsing
- **Statistics & inference** — configuration writers and batch workflows for PyCBC Inference,
  Bilby and LALInference; Fisher matrices; MCMC helpers
- **Visualization** — corner plots and analysis figures
- plus cosmology utilities, bundled detector noise curves, and ~30 command-line tools

## Installation

```bash
git clone https://github.com/gwnr/gwnr.git
cd gwnr
pip install -r requirements.txt
pip install .
```

See the [installation guide](https://gwnr.github.io/gwnr/docs/installation/) for details.

## Tutorials

 * [Making useful corner plots to visualize Bayesian posteriors](tutorials/MakingUsefulCornerPlots.html)
 * [Faithfulness between GW template models](tutorials/ComputeFaithfulness.html)
 * [Effectualness of GW search template banks](tutorials/ComputeEffectualness.html)
 * [Inference on GW events](tutorials/BayesianInferenceOnGWEvents.html)
 * [Inference on Injections](tutorials/BayesianInferenceOnGWInjections.html)
 * ... more [here](https://github.com/gwnr/gwnr/tree/master/tutorials)

## License

GNU General Public License — see [LICENSE](LICENSE).
