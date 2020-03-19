# Measuring neutron star tidal deformability with Advanced LIGO: A Bayesian analysis of neutron star-black hole binary observations

**Prayush Kumar<sup>1</sup>, Michael Purrer<sup>2</sup>, Harald P. Pfeiffer<sup>1,3,2</sup>**

**<sup>1</sup>1Canadian Institute for Theoretical Astrophysics, 60 St. George Street, University of Toronto, Toronto, ON M5S 3H8, Canada**

**<sup>2</sup>Max Planck Institute for Gravitational Physics (Albert Einstein Institute), Am Mühlenberg 1, 14476 Potsdam-Golm, Germany**

**<sup>3</sup>Canadian Institute for Advanced Research, 180 Dundas St. West, Toronto, ON M5G 1Z8, Canada**


## License

![Creative Commons License](https://i.creativecommons.org/l/by-sa/3.0/us/88x31.png "Creative Commons License")

This work is licensed under a [Creative Commons Attribution-ShareAlike 3.0 United States License](http://creativecommons.org/licenses/by-sa/3.0/us/).


## Introduction

This repository collects results published in [Kumar et al, 2017](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.95.044039).


The pioneering discovery of gravitational waves (GW) by Advanced LIGO has ushered us into an
era of observational GW astrophysics. Compact binaries remain the primary target sources for GW
observation, of which neutron star - black hole (NSBH) binaries form an important subset. GWs
from NSBH sources carry signatures of (a) the tidal distortion of the neutron star by its companion
black hole during inspiral, and (b) its potential tidal disruption near merger. In this paper, we
present a Bayesian study of the measurability of neutron star tidal deformability Λ_NS ∝ (R/M)^5
using observation(s) of inspiral-merger GW signals from disruptive NSBH coalescences, taking into
account the crucial effect of black hole spins. First, we find that if non-tidal templates are used to
estimate source parameters for an NSBH signal, the bias introduced in the estimation of non-tidal
physical parameters will only be significant for loud signals with signal-to-noise ratios greater than
~30. For similarly loud signals, we also find that we can begin to put interesting constraints on Λ_NS
(factor of 1 − 2) with individual observations. Next, we study how a population of realistic NSBH
detections will improve our measurement of neutron star tidal deformability. For an astrophysically
likely population of disruptive NSBH coalescences, we find that 20 − 35 events are sufficient to
constrain Λ_NS within ±25 − 50%, depending on the neutron star equation of state. For these
calculations we assume that LIGO will detect black holes with masses within the astrophysical
mass-gap. In case the mass-gap remains preserved in NSBHs detected by LIGO, we estimate that
approximately 25% additional detections will furnish comparable Λ_NS measurement accuracy. In
both cases, we find that it is the loudest 5 − 10 events that provide most of the tidal information,
and not the combination of tens of low-SNR events, thereby facilitating targeted numerical-GR
follow-ups of NSBHs. We find these results encouraging, and recommend that an effort to measure
Λ_NS be planned for upcoming NSBH observations with the LIGO-Virgo instruments.

## Notes

 * SEOBNRv2_ROM_DoubleSpin_LM contains a preliminary low-mass SEOBNRv2 DS ROM.
 * emcee_PE contains a Python PE code using emcee, lal and triangle.
 * emcee 
	  * Code: 	http://dan.iel.fm/emcee/current/ 
	  * Paper: 	http://arxiv.org/pdf/1202.3665.pdf
 * triangle
    * Code: 	https://github.com/dfm/triangle.py
