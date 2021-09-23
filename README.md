## Insitu_constrained_RF_SSM



<!-- ABOUT THE PROJECT -->
## About The Project
This study uses a random forest model to capture the highly non-linear relationship between the surface soil moisture and land surface features (and precipitation). In the end, to produce the long-term surface soil moisture at a global scale of 0.25 degrees. <br>

This repo contains the sciprt to produce the insitu constrained raindom forest surface soil moisture, including:

1. Obtaining the in-situ surface soil moisture from International Soil Moisture Network (ISMN), data is available on the official website: [https://ismn.geo.tuwien.ac.at/en/](https://ismn.geo.tuwien.ac.at/en/), the core package in this part of work: [https://pypi.org/project/ismn/](https://pypi.org/project/ismn/).

2. Downloading the land surface features from [Google Earth Engine (GEE)](http://code.earthengine.google.com/), including: 
> Land surface temperature [MOD11A1](https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MOD11A1)

> NDVI and EVI [MOD13A1](https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MOD13A1)

> Precipitation [ECMWF/ERA5](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_DAILY)

And synchronizing the land surface (/atmosphere) features with the in-situ SSM in spatial- and temporal- resolution (daily, 1km).

3. Training and testing the Random Forest Model with 70% of the data, and validating and evaluating with the rest 30%.

4. Applying the Trained RF model on the gridded land surface features to get the long-term in-situ contained global surface soil moisture.


## Acknowledgements
The author thanks R.Zhuang, Y.Zeng, B.szabo, S.Manfreda, Q.Han and Z.Su for their help with the result discussion.

## Contact
LZhang (leojayak@gmail.com)
