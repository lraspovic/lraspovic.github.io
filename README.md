**senL2Aproduct** and **senL2Aclass** are Python API-is used for searching, downloading, analysing and classifying Sentinel L2A satellite images that cover a specific area of interest.

## senL2Aproduct

Used for 2 sets of operations:
 - searching, previewing and downloading satellite images
 
 ```python
 from senL2Aproduct import SenL2Aproduct
 
 # Defining search parameters. Copernicus hub account is required.
 aoi_images = SenL2Aproduct(username='user', password='password', area_of_interest='aoi.shp', period='[YYYY-MM-DD TO YYYY-MM-DD]', cloudcover='[0 TO 5]')
 # Searching image(s) that completely cover aoi
 aoi_images.search_products(cover_aoi=True)
 # Previewing images
 aoi_images.show_products(page=1)
 # Downloading all images
 aoi_images.download_products(extract=True)
 
 ```
   
   *Preview example*
 <figure>
      <img src="https://user-images.githubusercontent.com/55063375/115270057-3be67200-a13c-11eb-8f59-48ccbc0082f3.png" alt="Preview_example" style="width:100%">
   </figure>
   
 - mozaic creation, clipping and stacking images

```python
from senL2Aproduct import clip_images, create_mosaic, stack_images
# Clipping images
clip_images(images='/images_folder', shape='aoi.shp', shape_crs=3857, bands=[2,3,4], resolution=['10m'])
# Creating mosaic
create_mosaic(bands=[2,3,4], folder_path='/clipped_images', level2A=False)
# Stack images
stack_images(bands=[2,3,4], folder_path='/mosaic', level2A=False)
```

   *Clip example*
  <figure>
      <img src="https://user-images.githubusercontent.com/55063375/115272148-5e798a80-a13e-11eb-9f87-cbd337b80b98.png" alt="Clip_example" style="width:100%">
   </figure>
  

## senL2Aclass

Used for automatic land cover classification of Sentinel L2A satellite images.

Two methods are used for classification:
- supervised classification - XGBoost
- unsupervised classification - ALCC (automatic land cover classification)

Images are classified in 6 classes:

  <figure>
      <img src="https://user-images.githubusercontent.com/55063375/115279075-866cec00-a146-11eb-8d10-cdee3fef23d0.png" alt="Classification_example" style="width:50%, height:50%">
   </figure>

```python
from senL2Aclass import ALCC, XGB_classification
# Define which Sentinel L2A image (product) to classify.
alcc_classification = ALCC(image='/SentinelL2Aimage.SAFE', level2A=True)
xgb_classification  = XGB_classification(image='/SentinelL2Aimage.SAFE', level2A=True)
# Run the classification algorithm
alcc_classification.run()
xgb_classification.run()
```

   *Classifications example*
  <figure>
      <img src="https://user-images.githubusercontent.com/55063375/115278372-a9e36700-a145-11eb-8601-98b6f5bb90c5.png" alt="Classification_example" style="width:100%">
   </figure>



All classifications are shown in web GIS created using Leaflet.js library.

https://lraspovic.github.io


