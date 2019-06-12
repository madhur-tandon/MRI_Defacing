# MRI_Defacing

## Motivation

- Increase in Research Interests to apply ML/DL to neuroscience
- Huge amounts of publicly datasets available
- Anonymization necessary to protect identity of test
subjects
- Also required by GDPR (General Data Protection
Regulation)
- Amount of defacing required to guarantee anonymity?
 
 ## Problem Statement
 
 - Build a binary classifier that can detect if an MRI Scan is
defaced or not.
    - Can serve as a lightweight deployable tool that researchers can use
as a check!
- Build a generative model that can learn to deface as well
as reface MRI Scans
    - Showcases that current methods of anonymization are partially
reversible to some extent
    - May not provide adequate protection to one’s identity
    
## Models Used

- Baseline
    - Logistic Regression
- Advanced
    - Kernelized SVM
    - Random Forests
    - CNN
- For Generation
    - Pix2Pix GAN
    
## Results for GAN

<img width="952" alt="Screenshot 2019-06-12 at 12 48 34 PM" src="https://user-images.githubusercontent.com/20173739/59330976-74cb8980-8d10-11e9-9e2c-565ed2e11cdb.png">

-------------------------------------

See [this](https://github.com/madhur-tandon/MRI_Defacing/blob/master/ML%20Project%20Main%20Slides.pdf) and [that](https://github.com/madhur-tandon/MRI_Defacing/blob/master/ML%20Project%20PPT.pdf) for additional information.
