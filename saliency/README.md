# Generating Saliency Masks

The following file describes two options for computing saliency masks. These form our pixel-grouping prior in the MaskContrast loss. 
The two options are as follows:
- __Option 1__: Use a supervised saliency detector. 
- __Option 2__: Use an unsupervised saliency detector.

We describe how to compute the saliency masks in both cases. Note that a postprocessing step is included. 

## Option 1: Supervised saliency

In this case, we simply use a saliency model pretrained in a supervised way to compute the saliency masks directly on the target dataset.
For our paper, we used the publicly available model from [BASNet](https://github.com/xuebinqin/BASNet).

```bibtex
@inproceedings{qin2019basnet,
  title={Basnet: Boundary-aware salient object detection},
  author={Qin, Xuebin and Zhang, Zichen and Huang, Chenyang and Gao, Chao and Dehghan, Masood and Jagersand, Martin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7479--7489},
  year={2019}
}
```

## Option 2: Unsupervised saliency

In this case, we do not use any datasets with annotated saliency masks.
We adopt a two-step strategy. In the first step, we use an unsupervised saliecny estimator to compute saliency masks on a publicly available saliency dataset.
We use the [DeepUSPS](https://github.com/sally20921/DeepUSPS) model. 
Note that the saliency datasets are more simpler in nature compared to segmentation datasets.
This allows to get high-quality masks even with an unsupervised model. 
In the second step, we use the obtained masks as pseudo ground-truth to train the BASNet model. 
The BASNet model is then used to compute saliency masks on the target dataset. 
We empirically found that this gave better results, compared to directly using the unsupervised model.
It seems that the BASNet architecture transfers better to new datasets compared to the model from DeepUSPS.
The weights of our BASNet model are publicly avaible on [google drive](https://drive.google.com/file/d/14qsoXU-NE63jKzuGPTJd8DRDnpnP4w6j/view?usp=sharing).

```bibtex
@inproceedings{nguyen2019deepusps,
  title={DeepUSPS: deep robust unsupervised saliency prediction with self-supervision},
  author={Nguyen, Duc Tam and Dax, Maximilian and Mummadi, Chaithanya Kumar and Ngo, Thi Phuong Nhung and Nguyen, Thi Hoai Phuong and Lou, Zhongyu and Brox, Thomas},
  booktitle={Proceedings of the 33rd International Conference on Neural Information Processing Systems},
  pages={204--214},
  year={2019}
}
```

## Postprocessing

The model predictions are post-processed using the method below.
This ensures that we do not have an image with many small segments. 

```python
import cv2
import numpy as np
from copy import deepcopy

def postprocess(model_output: np.array) -> np.array:
    """
	We postprocess the predicted saliency mask to remove very small segments. 
	If the mask is too small overall, we skip the image.

	Args:
	    model_output: The predicted saliency mask scaled between 0 and 1. 
	                  Shape is (height, width). 
	Return:
            result: The postprocessed saliency mask.
    """
	mask = (model_output > 0.5).astype(np.uint8)
	contours, _ = cv2.findContours(deepcopy(mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	
	# Throw out small segments
	for contour in contours:
	    segment_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
	    segment_mask = cv2.drawContours(segment_mask, [contour], 0, 255, thickness=cv2.FILLED)
	    area = (np.sum(segment_mask) / 255.0) / np.prod(segment_mask.shape)
            if area < 0.01:
		mask[segment_mask == 255] = 0

	# If area of mask is too small, return None
	if np.sum(mask) / np.prod(mask.shape) < 0.01:
	    return None

	return mask

```
