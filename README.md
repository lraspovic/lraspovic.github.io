**senL2Aproduct** and **senL2Aclass** are Python API-s used for searching, downloading and processing Sentinel L2A satellite images that cover specific area of interes.

## senL2Aproduct

Used for 2 sets of operations:
 - searching, previewing and downloading satelite images
   
   *Preview example*
 <figure>
      <img src="https://user-images.githubusercontent.com/55063375/115270057-3be67200-a13c-11eb-8f59-48ccbc0082f3.png" alt="Preview_example" style="width:100%">
   </figure>
   
 - mozaicing, clipping and stacking images

   *Clip example*
  <figure>
      <img src="https://user-images.githubusercontent.com/55063375/115272148-5e798a80-a13e-11eb-9f87-cbd337b80b98.png" alt="Clip_example" style="width:100%">
   </figure>
  

## senL2Aclass

Used for automatic classification of Sentinel L2A satellite images.

Two methods are used for classification:
- supervised classification - XGBoost
- unsupervised classification - ALCC (automatic land cover classification)

   *Classifications example*
  <figure>
      <img src="https://user-images.githubusercontent.com/55063375/115278372-a9e36700-a145-11eb-8601-98b6f5bb90c5.png" alt="Classification_example" style="width:100%">
   </figure>

Images are classified in 6 classes:

  <figure>
      <img src="https://user-images.githubusercontent.com/55063375/115279075-866cec00-a146-11eb-8d10-cdee3fef23d0.png" alt="Classification_example" style="width:50%, height:50%">
   </figure>


All classifications are shown in web gis created using Leaflet library.
https://lraspovic.github.io

This project was a part of master thesis.  
Entire thesis (in Croatian) can be found in the following link:
https://drive.google.com/file/d/1XA72yIkKAzVB2-j8irlLGd6FEZkXUymr/view?usp=sharing




