# KeySDL

KeySDL is a pipeline to fit GLV or replicator models of microbial systems from observations assumed to be at steady state. The research paper detailing the motivation, use cases, and validation of KeySDL is available as a [preprint].

This file contains the full scripts used in the KeySDL paper.
For easy use of KeySDL without modification, see the repository containing the minimal standalone implementation of KeySDL [here].

## Environment

The primary dependencies of the scripts in this repository are numpy, pandas, networkx, Seaborn, and (py)torch. In_Silico_SC.py also requires ryp and the R package miaSim [[3]](#3). Unless running the SOI model in In_Silico_SC.py it is recommended to omit these dependencies due to version incompatibility.

## File Descriptions

|File|Contents|
|---|---|
|carlstrom_reconstruction.py|Generates the results in the paper from data in [[1]](#1).|
|generate_glv_simulations.py|GLV model storage object with associated utility functions and SKLearn srapper for cross validation.|
|gutierrez_reconstruction.py|Generates the results in the paper from data in [[2]](#2).|
|In_Silico_GLV.py|Generates the In Silico GLV results in the paper.|
|In_Silico_Noise.py|Generates the noise characterization results in the paper.|
|In_Silico_SC.py|Generates the self-consistency scoring results in the paper. Requires an R environment with miaSim [[3]](#3).|
|plot_helpers.py|Plot generation utility functions.|
|reconstruct_from_ss.py|Reconstruction and self consistency functionality.|
|SparCC.py|Adaptation of SparCC from [[4]](#4) to Python3 for comparison with KeySDL.|

## References

<a id="1">[1]</a> 
Carlström, C.I., Field, C.M., Bortfeld-Miller, M., Müller, B., Sunagawa, S.,
Vorholt, J.A.: Synthetic microbiota reveal priority effects and keystone strains
in the arabidopsis phyllosphere. Nature Ecology & Evolution 3(10), 1445–1454, https://doi.org/10.1038/s41559-019-0994-z. 

<a id="2">[2]</a> 
Gutiérrez, N., Garrido, D.: Species deletions from microbiome consortia reveal
key metabolic interactions between gut microbes. mSystems 4(4), 00185–19, https://doi.org/10.1128/mSystems.00185-19.

<a id="3">[3]</a> 
Gao, Y., Şimşek, Y., Gheysen, E., Borman, T., Li, Y., Lahti, L., Faust, K., Garza,
D.R.: miaSim: an r/bioconductor package to easily simulate microbial community
dynamics. Methods in Ecology and Evolution 14(8), 1967–1980 https://doi.org/10.1111/2041-210X.14129.

<a id="4">[3]</a> 
Friedman, J., Alm, E.J.: Inferring correlation networks from genomic survey
data. PLoS Computational Biology 8(9), 1002687 (2012), https://doi.org/10.1371/journal.pcbi.1002687.
