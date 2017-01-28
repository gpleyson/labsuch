### Description
This is a python module contains an implementation of the modified Labusch
solute strengthening module, described in details in the links below.

### Module classes
**Primary classes**
- `LabuschParameters` - container for Labusch solute strengthening parameters
- `SoluteStrengtheningMinization` - class for calculating Labusch parameters
                                    from solute-dislocation interaction energies
- `SoluteStrengtheningModel` - class for predicting finite temperature yield
                               strengths of alloys

**Auxilliary classes and functions**
- `DislocationArray` - class for calculating the stress and displacement fields
                       of dislocation arrays.  Used for generating
                       solute-dislocation interaction energies of dissociated
                       dislocations (i.e. Peierls-Nabarro, spread-core)
- `SpreadCoreDislocation` - example of how to use the `DislocationArray` class
                            to generate interaction energies
- `displacement_edge_Volterra_ILE`, `displacement_screw_Volterra_ILE`,
  `stress_edge_Volterra_ILE`, `stress_screw_Volterra_ILE` - functions to
  generate the elastic fields of isotropic, linear elastic Volterra (perfect)
  dislocations


### Links:
- [NatMat 2010 - Quantitative prediction of solute strengthening in aluminium alloys ](http://www.nature.com/nmat/journal/v9/n9/full/nmat2813.html)
- [ActaMat 2012 - Solute strengthening from first principles and application to aluminum alloys ](http://www.sciencedirect.com/science/article/pii/S1359645412002273)
- [ActaMat 2012 First-principles prediction of yield stress for basal slip in Mg–Al alloys ](http://www.sciencedirect.com/science/article/pii/S1359645412003928)
- [PhilMag. 2013 - Friedel vs. Labusch: the strong/weak pinning transition in solute strengthened metals ](http://www.tandfonline.com/doi/abs/10.1080/14786435.2013.776718)
- [MSMSE 2016 - Solute strengthening at high temperatures ](http://iopscience.iop.org/article/10.1088/0965-0393/24/6/065005/meta;jsessionid=B628B8E99E30EDE55A0D199A575B889A.c3.iopscience.cld.iop.org)
