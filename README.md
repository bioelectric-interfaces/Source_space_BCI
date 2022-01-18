# Source_space_BCI

The code is structured in a way where for each decoder, we have a separate folder with files used to investigate the algorithm 

- ANN BCI - BCI that uses a chain of Convolutional Neural Networks to transform EEG data into predictions of ME task 
- Rhiemann - BCI that uses pyriemann library as a basis for decoder 
- CSP - BCI that uses Common Spatial Pattern approach to extract features to make classification 
- Source_estimation - BCI that performs transition to space of sources to decode ME task 

Also 
- Combine_results - code that was used to compare the intermediate results of the study 
- SG02 - all data gained from subject SG02, including MRI scan
