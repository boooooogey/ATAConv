# Description
The GitHub repository for totally interpretable sequence to function model (tiSFM), [publication](https://www.biorxiv.org/content/10.1101/2023.01.25.525572v1). tiSFM leverages prior information conveyed by the positive weight matrices (PWMs) of the known motifs via integrating PWMs as kernels to a convolutional layer. The model consists of popular, succesful techniques such as attention, convolution, positional embedding, and programmable pooling (a restricted version of an attention pooling). We have shown that the model's parameters correlates highly with the biological insight. Additionally, we compared tiSFM to AI-ATAC (current state of the art sequence to function model) and showed that tiSFM overperforms AI-ATAC.

![dark](https://user-images.githubusercontent.com/15932827/215548907-96246423-6aa6-4849-b42c-ad3502ba7bd0.png#gh-dark-mode-only)
![light](https://user-images.githubusercontent.com/15932827/215548553-7836e6bb-bcf9-4426-aa06-692b548dcfdd.png#gh-light-mode-only)

# Using this code

## Clone this repo

```
git clone https://github.com/boooooogey/ATAConv.git
```

## Run the code

```
python train_model.py --meme_file "Test.meme" --atac_file "ATACseqSignal.txt" --sequences "sequences.list" --cell_type "B" --model_output "test"
```
