# Similarity Estimation Network

This project was created as part of the automl exam 2019. The goal was to use methods presented in the lecture to outperform a baseline model or create our own method, which I did.

## Approach

The basic of my idea was to use the latent space or output of a neural network as a means to measure the similarities between known datasets and an unknown dataset, which could then be utilized as a metafeature to improve transferlearning.
Transferlearning is using a model pretrained on data from a similar domain, and optionally their hyperparameters, to reduce the resources needed to train a model to satisfactory performance from scratch.
With K49 as the target I built a portfolio of pretrained restnet18s for EMNIST, KMNIST, MNIST, OMNIGLOT, and QMNIST. A configuration option was added to BOHB's configuration space to used the pretrained weight from the dataset with the most similarities, in short warmstarting the training.
The similarities were estimated by using the symmetric difference in the normalized output distribution of the unknown and known dataset fed through the model trained on omniglot. The sen, similarity estimation network, was trained on omniglot to later measure the similarity based on the wide variety of characters it contains from different languages.

## Results

Well, the idea was nice but mistakes were made on a technical and conceptual level. At least the symmetric difference between KMNIST and the target was expectedly small.

## Discussion

The warmstarting was mistakenly dependent on the use of data augmenting transformations. Because the output distribution of the sen was close to the label distribution, yay for high accuracy, the symmetric difference to the target dataset was very high compared to the difference to KMNIST. When comparing the training loss during the first 8 epochs, random initialization and the model pretrained on omniglot outperformed all other pretrained networks.\
To avoide such obvious flaw in the future, it might be interesting to create a multi modal vae that could link and encode different type of data like video, text, pictures, etc. with powerful latent space as the metafeature to measure similarities between tasks allowing for utilizing pretrained models. Additionally, a sampling strategy could be introduced to avoid computing the metafeature for all traget data.

## Sidenotes

### AutotuneLr

⚠️*still needs verification*⚠️\
Using refined gridsearch, a possibly optimal initial learning rate is determined, which allows to remove it from the configspace used by BOHB.

### BSGuard

⚠️*still needs verification*⚠️\
A decorator allowing to recover from cuda's out of memory exceptions which can be used to adjust the batchsize without restarting the training.
In combination with AutotuneLR, available vram, and model size, the batchsize can be maximized, reducing wasted resources.
