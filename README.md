# Description
The GitHub repository for totally interpretable sequence to function model (tiSFM), [publication](https://www.biorxiv.org/content/10.1101/2023.01.25.525572v1). tiSFM leverages prior information conveyed by the positive weight matrices (PWMs) of the known motifs via integrating PWMs as kernels to a convolutional layer. The model consists of popular, succesful techniques such as attention, convolution, positional embedding, and programmable pooling (a restricted version of an attention pooling). We have shown that the model's parameters correlates highly with the biological insight. Additionally, we compared tiSFM to AI-ATAC (current state of the art sequence to function model) and showed that tiSFM overperforms AI-ATAC.

The model architecture:

<p align="center">
  <img src="https://user-images.githubusercontent.com/15932827/215568629-ccea3edc-556c-43b9-a5e3-6307f844c12c.png">
</p>

Comparison with AI-ATAC:

![dark](https://user-images.githubusercontent.com/15932827/215548907-96246423-6aa6-4849-b42c-ad3502ba7bd0.png#gh-dark-mode-only)
![light](https://user-images.githubusercontent.com/15932827/215548553-7836e6bb-bcf9-4426-aa06-692b548dcfdd.png#gh-light-mode-only)

# Training a model
To train a model, we provide ``train_model.py``.

* ``--meme-file``
  
  The MEME file that includes the PWMs for the motifs. Excerpt from a MEME file:
```
MEME version 4

ALPHABET= ACGT

strands: + -

Background letter frequencies
A 0.25 C 0.25 G 0.25 T 0.25

MOTIF Nobox+M1945_1.02+D
letter-probability matrix: alength= 4 w= 8 nsites= 1 E= 0
 0 0.0263157894736839 0 0.973684210526316
 0.947368421052632 0 0 0.0526315789473681
 1 0 0 0
 0.0263157894736841 0.0263157894736841 0.0526315789473682 0.894736842105264
 0.0526315789473681 0 0.10526315789474 0.842105263157892
 0.394736842105262 0 0.578947368421054 0.0263157894736842
 0.105263157894739 0.315789473684207 0.473684210526315 0.105263157894739
 0.0526315789473681 0.342105263157891 0.15789473684211 0.447368421052631
```
* ``--atac-file``

  A tab-separated file that contains the signal (aggregated zscored ATAC-seq signal for example) to be predicted. A small example:
```
        abT     B       gdT     innate.lym      myeloid stem    stroma  T.act
ImmGenATAC1219.peak_1   -0.130933631917673      0.138651166389695       0.156971008389084       0.527077542143752       -0.293308497356562      -0.288624780390918      0.266461045986036       -0.135626004100113
ImmGenATAC1219.peak_2   0.215899772687732       -0.222235583621337      0.0129926719125442      -0.287304588995824      -0.279635332939133      0.178730565913627       0.821688546905942       0.407705244680555
ImmGenATAC1219.peak_3   -0.230929495003157      0.0509345422484776      -0.224679261877864      -0.0236462069497591     -0.249849003896584      0.238851736127574       1.56834498814214        0.0016550477655455
ImmGenATAC1219.peak_4   -0.0942398935366446     0.0781131057646563      -0.610750492562313      0.321371781313088       -0.223094333117685      -0.508111698135677      0.905709354510175       0.324178907569005
ImmGenATAC1219.peak_5   0.120764205927777       -0.10946201054957       0.381757877407664       -0.35943594076623       0.131556817588826       -0.40623779125231       0.910415764975103       -0.383010002546782
```
* ``--sequences``
  
  A file that contains the sequences extracted from the peak regions. The rows of ``atac-file`` and ``sequences`` should match.
```
ImmGenATAC1219.peak_1   tttcccagaagctgtgttgctttggccagggaggtggctggttgtctggagccgaaaatggtgccacctcagaaggtctctggctctcgcctgtcccagaaactgctggcctctgtattccacaccctcacccatgcagcctgccctcctcagagtccggaaccaaggtggctcctgcggagcctgaggcagaaacctcttgggccgggtggacccctgtgctctcaccaggaaggtggccggttgtctgtagccgaaaatggcaccacctcagaagctctgtggctcttgcctgtccca
ImmGenATAC1219.peak_2   ctggaaacccccatcccatcctccccttgcttctataatggtgctcctccactcacccacacactcccaccttctcactctccagtcccctacactgtatcatctatccagctttcataggaccaaggacctctcccattgatgcctgacaaggccaacatctgctacatatacagctggagccatgtatactccttgattggtggcttagtccctggaggctcgggggggggtggggttggttggttgatattattgttcttatggggttgcaaaccctttcaactttttcagtcctgt
ImmGenATAC1219.peak_3   GGAGAATGAGATGTCCCTGGAGAAATTGTTTTCTTAGCCAGTGGCTGCCAGTTTTAGAGAAAACTATGTTTGTCTCCACAGTGACGGAGCAGGAGTGTGCAAGAAAACCCTCTGAGGGTTTGCTTGGATAACAAACATTATGCATCATGGGTCCCAAGTGCATACTTACTTAATGCCAAAAATTTGAAAAATATTTCCAGACAAGTAAGAAGAGGCAGATAGCTTATAAATTAAAATGAATGAATATTAACTGTGGGTGAAATTCAGAGCCTGGGTTTATTTTCCAAGGGCTATATTTGA
ImmGenATAC1219.peak_4   atttaaagcTAGCTTAGACTTTATAACAAAAGCCCACTCACAAGACAAGACCAGGATCCTGGAGAGATAGTTCAGGAAGCACAGACTCTTGCTATGGGCTTATAAGTCTCCATGTCTGCATCTCCAGGGAAAAGCTGAGAGTCAGCATGAATCCAGAGATAGAAAAAGTGGCAGGACTTTCTGGCTGCCAGCCCAGAATCAAGTTGAATGAGAGAAACTTCCTCAAAGGAATAAGGCAGAGAATGATGTTCCGGCTTCCATGTGTTCCTGGGCCTGTAGTCTCTCAAACACATCtgttgt
ImmGenATAC1219.peak_5   GTATACTCTTTTAGAAGCTACTTTTGGACCTACCCTCTGCCCTTCCTTTTTTTCCAAGACATCAGCAGGCTGGTTGAGAGCGTTAATGGTCCCAACCAATGAGGAACTAGCTGTGAGGAGAGCAGTCAAGTGTGTCCTGAAGTTCAGATGGCTTTCTGCAACCAGATACTTGCCAGACATGAGCAGTAAACACAGTTCCATCAGTCCAGTTCCACACAAAGACTGAGAGCATGATTCAGCCCTCTGTCTGCTTCATTTGCCTTACAGTTCACTTGGTTGATCATTACCTTGAATGTCATT
```
* ``--batch-size``

  Specify the batch size for the training. 
* ``--model-output``

  Specify the path to save the model parameters.
* ``--split-folder``

  A directory that contains three files: ``train_split.npy``, ``validation_split.npy``, ``test_split.npy``, each a Numpy integer array that contain row indices for the data set.
* ``--number-of-epochs``

  Specify how many times the training goes over a single sample in the training split. 
* ``--lr``

  Initial learning rate. The learning rate decays if the MSE on the validation split does not improve for 10 epochs.
* ``--early-stopping``

  Set to enable early stopping. The training stops before all the epochs are iterated if the learning rate decays below a threshold.
* ``--early-stopping-threshold``

  Set the learning rate threshold for early stopping.
* ``--penalty-type``

  Specify through which proximal operator to enfore sparsity on the model's parameters. The options are ``mcp`` and ``l1``.
* ``--penalty-param``

  The hyperparameter for ``l1`` and ``mcp``. In the case of ``mcp``, this is the first hyperparameter.
* ``--mcp-param``

  The second hyperparameter for ``mcp`` proximal operator.
* ``--penalty-param-range``

  Three values for ``start``, ``end``, and ``step`` for the hyperparameter range of the path algorithm.
* ``--architecture``

  The architecture to be used. When ``--architecture mode_name`` is given, the script looks ``model/mode_name.py`` to import the model architecture from. Usually the models in ``model`` directory inherit the template model (``TemplateModel``) in ``template_model.py`` which includes the convolutional layer with PWM and a final layer that predicts the target signals.

## Clone this repo

```
git clone https://github.com/boooooogey/ATAConv.git
```

## Run the code

```
python train_model.py --architecture model_pos_attention_calib_sigmoid_interaction --meme-file cisBP_mouse.meme --atac-file lineageImmgenDataZ.txt --sequences sequences.list --model-output checkpoint --split-folder split_folder --batch-size 254
```
