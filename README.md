# COMP5329Assignment2
## Set up
you need to create two empty dirs for the input and the output, name them "Input" and "Output" in "Code", then run ``py predict.py``

## Inport your own model
We provide a default model for this net, however, if you are unsatisfied by the performance of the default model, you can import your own model simply put it in the dir "Algorithm", then go to ``CONFIG.py`` and change the ``MODEL`` and  ``LABEL``

## Performance
The default model is trained with only 10000 images, you can retrain it if you want to

## Net
We use transfer learning to simplify the assignment, only 3 FC layers are added to the DenseNet, the DenseNet itself is untouched

