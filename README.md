# Transformers_two_encoders
A two layer encoder transformer with slight modifications allowing for masking and padding. Applicable to multivariate time series data. Created so I could reference it but public since others could use it as well. :)

Dataset is not included due to confidentilaity issues.

![Model Image](images/transformer_model_for_paper.png)
The code is meant to create a transformer architecture with a custom multihead attention factoring for masking as shown above. This transformer has two encoders as shown in image.

All package files that produce this code is provided in requirements.txt. The code also does interpolation which is not fully resolved. Please reference the 'transformer_no_interpolation.py' for code which does work. Additionally, that file contains code which generates useful figures as shown in the images folder. The code also prints important information like RAM and time it takes for training. Finally, the code creates a way to save the model and scalar and reinference it at the bottom of 'transformer_no_interpolation.py'
