# Standard library
import os
import operator
import time

# General imports
import numpy as np
from sklearn import cluster
import xgboost as xgb

# Spatial imports
import rasterio
from rasterio.enums import Resampling
from rasterio import features
from rasterio.plot import reshape_as_image
from rasterio import Affine as A
from rasterio.merge import merge
from shapely.geometry import shape, Polygon, mapping
import fiona
import geopandas as gpd


import matplotlib.pyplot as plt


np.seterr(divide='ignore', invalid='ignore')


class ALCC():

    '''Class for Automatic Land Cover Classification of Sentinel Level 2A products.
    Classification algorithm is based on vegetation indices, which serve as a input for unsupervised classification algorithm - K means.
    K means in combination with tresholding classification create final land cover raster.

    Parameters
    ----------
    image : string
        Path to Sentinel Level 2A product (unzipped folder) or multiband satellite image. Order of bands in multiband image must be in following order:
        2 - Blue
        3 - Green
        4 - Red
        8 - NIR
        11 - SWIR 1
        12 - SWIR 2
        SCL - Scene classification raster

    level2A : bool
        If "image" is a path to Sentinel Level 2A product, must be set to True. Defaults to True.
    '''

    def __init__(self, image, level2A=True):
        self.image = image
        self.level2A = level2A
        self.images_10m = []
        self.images_to_resample = []
        self.scene_classification = None

    def set_file_paths(self):

        if self.level2A:
            for dirpath, dirnames, filenames in os.walk(self.image):
                if dirpath.endswith('R10m'):
                    for image in os.listdir(os.path.join(dirpath)):
                        if any(band in image for band in ['B02', 'B03', 'B04', 'B08']):
                            self.images_10m.append(
                                os.path.join(dirpath, image))
                elif dirpath.endswith('R20m'):
                    for image in os.listdir(dirpath):
                        if 'B11' in image or 'B12' in image:
                            self.images_to_resample.append(
                                os.path.join(dirpath, image))
                        elif 'SCL' in image:
                            self.scene_classification = os.path.join(
                                dirpath, image)

    def _create_folder(self, name):
        # Creates folder in current directory
        root = self._get_root()
        new_dir = os.path.join(root, name)
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        return new_dir

    @staticmethod
    def _get_root():
        return os.path.abspath(os.path.dirname(__file__))

    def _get_image_path(self, folder, image_to_find):
        for image in os.listdir(self._create_folder(folder)):
            if image_to_find in image:
                return os.path.join(self._create_folder(folder), image)

    def resample_image(self, res='10m', names=None):
        '''Resamples bands to target resolution. Method used for resampling is bilinear.

        Parameters
        ----------

        res : str
            Target resolution in meters.

        names : list
            Names to give resampled images.
        '''

        print('Resampling images...')

        resampled_dir = self._create_folder('resampled')

        name_len = len(names)

        for rsp_img in os.listdir(resampled_dir):
            for name in names:
                if name in rsp_img:
                    name_len -= 1

        if name_len == 0:
            print('All images already resampled')
            self.scene_classification = os.path.join(
                resampled_dir, names[2]) + ".tif"
            return

        for dirpath, dirnames, filenames in os.walk(self.image):
            if dirpath.endswith(res):
                for image in os.listdir(dirpath):
                    sample_image = os.path.join(dirpath, image)
                    break

        with rasterio.open(sample_image) as sample_src:
            target_height = sample_src.height
            target_width = sample_src.width
            meta = sample_src.profile

        meta.update(
            driver='Gtiff'
        )

        for index, image in enumerate(self.images_to_resample):
            with rasterio.open(image) as src:
                img = src.read(
                    out_shape=(
                        src.count,
                        target_height,
                        target_width
                    ),
                    resampling=Resampling.bilinear
                )

            with rasterio.open(os.path.join(resampled_dir, names[index]) + ".tif", "w", **meta) as dst:
                dst.write(img)

        with rasterio.open(self.scene_classification) as src:
            img = src.read(
                out_shape=(
                    src.count,
                    target_height,
                    target_width
                ),
                resampling=Resampling.bilinear
            )
        meta.update(
            dtype='uint8'
        )

        with rasterio.open(os.path.join(resampled_dir, names[2]) + ".tif", 'w', **meta) as dst:
            dst.write(img)

        self.scene_classification = os.path.join(
            resampled_dir, names[2]) + ".tif"

    def calculate_vi(self, v_indices):
        '''Calculates vegetation indices

        Parameters
        ----------

        v_indices : object
            List of vegetation indices to calculate.

        '''

        print('Calculating vi...')

        veg_index_dir = self._create_folder('veg_index')

        if self.level2A:

            resampled_dir = self._create_folder('resampled')

            bands_for_veg_index = self.images_10m

            for image in os.listdir(resampled_dir):
                bands_for_veg_index.append(os.path.join(resampled_dir, image))

        for v_index in v_indices:

            # NDVI = (NIR - RED) / (NIR + RED)
            if v_index == 'NDVI':

                if os.path.exists(os.path.join(veg_index_dir, 'NDVI.tif')):
                    continue

                if self.level2A:
                    for band in bands_for_veg_index:
                        if 'B04' in band:
                            red_p = band
                        elif 'B08' in band:
                            nir_p = band

                    with rasterio.open(red_p) as red_src:
                        red = red_src.read(1).astype('float32')
                        meta = red_src.meta

                    with rasterio.open(nir_p) as nir_src:
                        nir = nir_src.read(1).astype('float32')
                else:
                    with rasterio.open(self.image) as src:
                        red = src.read(3).astype('float32')
                        nir = src.read(4).astype('float32')
                        meta = src.meta

                    meta.update(
                        count=1,
                    )

                ndvi = ((nir - red) / (nir + red)).astype('float32')
                ndvi = np.nan_to_num(ndvi)
                ndvi = np.where(((ndvi > 1) | (ndvi < -1)), 0, ndvi)

                meta.update(
                    driver='Gtiff',
                    dtype='float32',
                    nodata=0
                )

                with rasterio.open(os.path.join(veg_index_dir, 'NDVI.tif'), 'w', **meta) as dst:
                    dst.write(ndvi, 1)

            # MNDWI = (GREEN − SWIR 1) / (GREEN + SWIR 1)
            elif v_index == 'MNDWI':

                if os.path.exists(os.path.join(veg_index_dir, 'MNDWI.tif')):
                    continue

                if self.level2A:

                    for band in bands_for_veg_index:
                        if 'B03' in band:
                            green_p = band
                        elif 'SWIR1' in band:
                            swir1_p = band

                    with rasterio.open(green_p) as green_src:
                        green = green_src.read(1).astype('float32')
                        meta = green_src.meta

                    with rasterio.open(swir1_p) as swir1_src:
                        swir1 = swir1_src.read(1).astype('float32')

                else:
                    with rasterio.open(self.image) as src:
                        green = src.read(2).astype('float32')
                        swir1 = src.read(5).astype('float32')
                        meta = src.meta

                    meta.update(
                        count=1,
                    )

                mndwi = ((green - swir1) / (green + swir1)).astype('float32')
                mndwi = np.nan_to_num(mndwi)
                mndwi = np.where(((mndwi > 1) | (mndwi < -1)), 0, mndwi)

                meta.update(
                    driver='Gtiff',
                    dtype='float32',
                    nodata=0
                )

                with rasterio.open(os.path.join(veg_index_dir, 'MNDWI.tif'), 'w', **meta) as dst:
                    dst.write(mndwi, 1)

            # NDTI = (SWIR 1 − SWIR 2) / (SWIR 1 + SWIR 2)
            if v_index == 'NDTI':

                if os.path.exists(os.path.join(veg_index_dir, 'NDTI.tif')):
                    continue

                if self.level2A:

                    for band in bands_for_veg_index:
                        if 'SWIR1' in band:
                            swir1_p = band
                        elif 'SWIR2' in band:
                            swir2_p = band

                    with rasterio.open(swir1_p) as swir1_src:
                        swir1 = swir1_src.read(1).astype('float32')
                        meta = swir1_src.profile

                    with rasterio.open(swir2_p) as swir2_src:
                        swir2 = swir2_src.read(1).astype('float32')

                else:
                    with rasterio.open(self.image) as src:
                        swir1 = src.read(5).astype('float32')
                        swir2 = src.read(6).astype('float32')
                        meta = src.meta

                    meta.update(
                        count=1,
                    )

                ndti = ((swir1 - swir2) / (swir1 + swir2)).astype('float32')
                ndti = np.nan_to_num(ndti)
                ndti = np.where(((ndti > 1) | (ndti < -1)), 0, ndti)

                meta.update(
                    driver='Gtiff',
                    dtype='float32',
                    nodata=0
                )

                with rasterio.open(os.path.join(veg_index_dir, 'NDTI.tif'), 'w', **meta) as dst:
                    dst.write(ndti, 1)

            # BAEI = (RED + 0.3) / (GREEN + SWIR)
            elif v_index == 'BAEI':

                if os.path.exists(os.path.join(veg_index_dir, 'BAEI.tif')):
                    continue

                if self.level2A:

                    for band in bands_for_veg_index:
                        if 'B03' in band:
                            green_p = band
                        elif 'B04' in band:
                            red_p = band
                        elif 'SWIR1' in band:
                            swir1_p = band

                    with rasterio.open(green_p) as green_src:
                        green = green_src.read(1).astype('float32')
                        meta = green_src.meta

                    with rasterio.open(red_p) as red_src:
                        red = red_src.read(1).astype('float32')

                    with rasterio.open(swir1_p) as swir1_src:
                        swir1 = swir1_src.read(1).astype('float32')

                else:
                    with rasterio.open(self.image) as src:
                        green = src.read(2).astype('float32')
                        red = src.read(3).astype('float32')
                        swir1 = src.read(5).astype('float32')
                        meta = src.meta

                    meta.update(
                        count=1,
                    )

                baei = (red + 0.3) / (green + swir1).astype('float32')
                baei = np.nan_to_num(baei)
                baei = np.where(((baei > 1) | (baei < -1)), 0, baei)

                meta.update(
                    driver='Gtiff',
                    dtype='float32',
                    nodata=0
                )

                with rasterio.open(os.path.join(veg_index_dir, 'BAEI.tif'), 'w', **meta) as dst:
                    dst.write(baei, 1)

    def mask_clouds(self):
        '''
        Extracts clouds, shadows, ice and dark area pixels from 2A scene classification.

        '''

        print('Masking clouds...')

        mndwi_p = self._get_image_path('veg_index', 'MNDWI')

        with rasterio.open(mndwi_p) as src:
            mndwi = src.read(1).astype('float32')
            meta = src.profile
            transform = src.transform
            crs = src.crs

        if self.level2A:

            with rasterio.open(self.scene_classification) as scl_src:
                scl = scl_src.read(1).astype('uint16')

        else:
            with rasterio.open(self.image) as scl_src:
                scl = scl_src.read(7).astype('uint16')

        dark_area = 2

        cloud_shadows = 3

        cloud_medium = 8

        cloud_high = 9

        snow_pix = 11

        # True False mask
        cloud_mask = (scl == cloud_medium) | (scl == cloud_high)

        buffered = False

        classes_dir = self._create_folder('classes')

        for classes in os.listdir(classes_dir):
            if 'Clouds' in classes:
                buffered = True
                with rasterio.open(os.path.join(classes_dir, classes)) as src:
                    clouds_buff = src.read(1)

        if not buffered:
            # buffered cloud raster
            clouds_buff = self._buffer_and_save_clouds(
                scl, cloud_mask, transform, crs, mndwi_p)

        # no_buffer cloud raster
        cloud_mask = np.where(cloud_mask, 1, 0)

        if clouds_buff is not None:
            cloud_mask_combined = np.where(
                (clouds_buff + cloud_mask) >= 1, 0, 1)
        else:
            cloud_mask_combined = np.where((cloud_mask) >= 1, 0, 1)
        clouds = np.where(cloud_mask_combined == 0, 6, 0).astype('uint8')

        shadow_mask = np.where(
            ((scl == dark_area) | (scl == cloud_shadows)), 0, 1)
        shadows = (np.where(shadow_mask == 0, 7, 0) * cloud_mask_combined)

        snow_mask = np.where(scl == snow_pix, 0, 1)
        snow = (np.where(snow_mask == 0, 8, 0) *
                cloud_mask_combined).astype('uint8')

        mndwi_masked = np.where(
            (mndwi * cloud_mask_combined * snow_mask) == 0, 99999, mndwi).astype('float32')

        masked_dir = self._create_folder('masked')

        meta.update(
            nodata=99999
        )

        with rasterio.open(os.path.join(masked_dir, 'MNDWI_no_clouds.tif'), 'w', **meta) as dst:
            if len(mndwi_masked.shape) == 3:
                dst.write(mndwi_masked)
            else:
                dst.write(mndwi_masked, 1)

        meta.update(
            nodata=0,
            dtype='uint8'
        )

        with rasterio.open(os.path.join(classes_dir, 'Clouds.tif'), 'w', **meta) as dst:
            if len(clouds.shape) == 3:
                dst.write(clouds)
            else:
                dst.write(clouds, 1)

        with rasterio.open(os.path.join(classes_dir, 'Snow.tif'), 'w', **meta) as dst:
            if len(snow.shape) == 3:
                dst.write(snow)
            else:
                dst.write(snow, 1)

    def _buffer_and_save_clouds(self, scl, mask, transform, crs, raster_p):

        print('Buffering clouds...')

        vector_dir = self._create_folder('vector')

        classes_dir = self._create_folder('classes')

        vectorized = False

        for vector in os.listdir(vector_dir):
            if 'clouds_buff.shp' in vector:
                vectorized = True
                clouds = gpd.read_file('clouds_buff.shp')

        if not vectorized:

            shapes = features.shapes(scl, mask=mask, transform=transform)

            buffer_polygons = []

            for g, v in shapes:

                for index, tup in enumerate(g['coordinates'][0]):
                    g['coordinates'][0][index] = list(tup)

                poligon = Polygon(g['coordinates'][0])

                # Buffer only big clouds
                if poligon.area >= 100000:

                    poligon_buffer = poligon.buffer(
                        200, cap_style=3, join_style=2)

                    buffer_polygons.append(poligon_buffer)

            if len(buffer_polygons) == 0:
                print('No big clouds')
                return None

            schema = {
                'geometry': 'Polygon',
                'properties': {'id': 'int'},
            }

            with fiona.open(os.path.join(vector_dir, 'clouds_buff.shp'), 'w', 'ESRI Shapefile', schema=schema, crs=crs) as shp:

                for buff in buffer_polygons:
                    shp.write({
                        'geometry': mapping(buff),
                        'properties': {'id': 6}
                    })

            clouds = gpd.read_file(os.path.join(vector_dir, 'clouds_buff.shp'))

        with rasterio.open(raster_p) as src:
            meta = src.meta

        meta.update(nodata=0)

        with rasterio.open(os.path.join(classes_dir, 'Clouds.tif'), 'w+', **meta) as out:
            out_arr = out.read(1)

            shapes = ((geom, value)
                      for geom, value in zip(clouds.geometry, clouds.id))

            burned = features.rasterize(
                shapes=shapes, fill=0, out=out_arr, transform=out.transform)
            out.write_band(1, burned)

            return out.read(1).astype('uint8')

    def k_means_classification(self, index, k, masked=False):
        '''Performs K means classification on vegetation index.

        index : string
            Path to a vegetation index.

        k : int
            Number of classes.

        masked : bool
            Defines if vegetation index is masked or not. Defaults to False.

        '''

        if not masked:
            k_means_index = self._get_image_path('veg_index', index)

        else:
            k_means_index = self._get_image_path('masked', index)

        k_means_dir = self._create_folder('k_class')

        with rasterio.open(k_means_index) as src:
            img = reshape_as_image(src.read())
            meta = src.meta

        meta.update(
            dtype='int8',
            nodata=-1
        )

        X = img.reshape((-1, img.shape[2]))
        k_means = cluster.MiniBatchKMeans(n_clusters=k)
        print('Kmeans classification...')
        k_means.fit(X)

        X_cluster = k_means.labels_
        X_cluster = X_cluster.reshape(img[:, :, 0].shape).astype('int8')

        with rasterio.open(os.path.join(k_means_dir, str(index) + "_K" + str(k) + ".tif"), 'w', **meta) as dst:
            dst.write(X_cluster, 1)

    def masking_image(self, index, mask_index=None):
        '''
        Masks vegetation index by masked index

        Paramaters
        ----------

        index : string
            Path to a vegetation index.

        mask_index : string
            Path to a masked vegetation index.
        '''

        print('Masking vi...')

        # Get vegetation index to mask
        v_index_p = self._get_image_path('veg_index', index)
        with rasterio.open(v_index_p) as src:
            v_index = src.read(1)
            meta = src.meta

        # Get vegetation index that was classified = mask index
        m_index_p = self._get_image_path('masked', mask_index)
        with rasterio.open(m_index_p) as src:
            m_index = src.read(1)

        # Get classification of mask index
        k_class_p = self._get_image_path('k_class', mask_index)
        with rasterio.open(k_class_p) as src:
            k_class = src.read(1)

        meta.update(
            dtype='uint8',
            nodata=0
        )

        if mask_index == 'MNDWI':

            water, other = self.extract_classes(
                mask_index, m_index, k_class, path=False)

            with rasterio.open(os.path.join(self._create_folder('classes'), 'Water.tif'), 'w', **meta) as dst:
                dst.write(water, 1)

        elif mask_index == 'NDVI':

            high_veg, low_veg, other = self.extract_classes(
                mask_index, m_index, k_class, path=False)

            with rasterio.open(os.path.join(self._create_folder('classes'), 'High_veg.tif'), 'w', **meta) as dst:
                dst.write(high_veg, 1)

            with rasterio.open(os.path.join(self._create_folder('classes'), 'Low_veg.tif'), 'w', **meta) as dst:
                dst.write(low_veg, 1)

        vi_mask = np.where(v_index * other == 0, 99999,
                           v_index).astype('float32')

        # os.chdir(os.path.join(root, 'masked'))

        meta.update(
            dtype='float32',
            nodata=99999
        )

        other_name = index + '__other.tif'

        with rasterio.open(os.path.join(self._create_folder('masked'), other_name), 'w', **meta) as dst:
            dst.write(vi_mask, 1)

    def extract_classes(self, name, index, index_class, path=True, masked=False):
        '''
        Extracts classes from k means classification of vegetation index.

        Parameters
        ----------

        name : string
            Vegetation index used to extract land cover classes.

        index : string
            Vegetation index file name

        index_class : string
            Vegetation index k means classification name

        path : bool
            Defines if path to a file is sent for "index" and "index_class". If False must send numpy array for both "index" and "index_class".

        masked : string
            Defines if the path is to the masked index. Defaults to False.

        '''

        print('Extracting classes...')

        if path:
            if not masked:
                v_index_p = self._get_image_path('veg_index', index)
            else:
                v_index_p = self._get_image_path('masked', index)

            with rasterio.open(v_index_p) as src:
                v_index = src.read(1)
                meta = src.meta

            k_class_p = self._get_image_path('k_class', index_class)
            with rasterio.open(k_class_p) as src:
                k_class = src.read(1)
        else:

            v_index = index
            k_class = index_class

        cluster_means = {}

        for cluster_val in np.unique(k_class):
            cluster_arr = np.where(k_class == cluster_val, 1, 0)
            mask = v_index * cluster_arr
            if mask[mask > 1].size > 0:
                continue
            mask_nan = np.where(mask == 0, np.nan, mask)
            mean = np.nanmean(mask_nan)
            cluster_means.update({mean: cluster_arr})

        cluster_means_sorted = dict(
            sorted(cluster_means.items(), key=operator.itemgetter(0), reverse=True))

        if name == 'MNDWI':

            # Checks if there is a water class
            if list(cluster_means_sorted.keys())[0] < 0:
                print('No water classified')
                print('Extracting water from index')
                water = np.where(
                    ((v_index < 1) & (v_index > 0.4)), 1, 0).astype('uint8')
                other = np.where(v_index <= 0.4, 1, 0).astype('uint8')
            else:
                water = cluster_means_sorted.get(
                    list(cluster_means_sorted.keys())[0]).astype('uint8')
                other = cluster_means_sorted.get(
                    list(cluster_means_sorted.keys())[1]).astype('uint8')
            return (water, other)

        elif name == 'NDVI':
            high_veg = cluster_means_sorted.get(
                list(cluster_means_sorted.keys())[0]).astype('uint8') * 3
            low_veg = cluster_means_sorted.get(
                list(cluster_means_sorted.keys())[1]).astype('uint8') * 2
            other = cluster_means_sorted.get(
                list(cluster_means_sorted.keys())[2]).astype('uint8')
            return (high_veg, low_veg, other)

        elif name == 'NDTI':
            meta.update(
                dtype='uint8',
                nodata=0
            )
            num_of_class = len(cluster_means_sorted)
            if num_of_class == 3:
                soil = (cluster_means_sorted.get(list(cluster_means_sorted.keys())[0]) +
                        cluster_means_sorted.get(list(cluster_means_sorted.keys())[1])).astype('uint8') * 4
                built_up = cluster_means_sorted.get(
                    list(cluster_means_sorted.keys())[-1]).astype('uint8') * 5

            elif num_of_class == 2:
                print('Could not classify into 3 classes. \n Classified into 2 classes')
                soil = cluster_means_sorted.get(
                    list(cluster_means_sorted.keys())[0]).astype('uint8') * 4
                built_up = cluster_means_sorted.get(
                    list(cluster_means_sorted.keys())[-1]).astype('uint8') * 5

            else:
                print('Could not separate.')
                soil = cluster_means_sorted.get(
                    list(cluster_means_sorted.keys())[0]).astype('uint8') * 4
                with rasterio.open(os.path.join(self._create_folder('classes'), 'Soil_NDTI_2.tif'), 'w', **meta) as dst:
                    dst.write(soil, 1)
                return

            with rasterio.open(os.path.join(self._create_folder('classes'), 'Soil_NDTI_km.tif'), 'w', **meta) as dst:
                dst.write(soil, 1)

            with rasterio.open(os.path.join(self._create_folder('classes'), 'Built_up_NDTI_km.tif'), 'w', **meta) as dst:
                dst.write(built_up, 1)

    def tresholding_classification(self, index, tresh_value, masked=False):
        '''
        Classifies vegetation index by treshold values.

        Paramaters
        ----------

        index : string
            Vegetation index to treshold.

        tresh_value : object
            List of 2 values used as treshold values. First one represents minimum and the second maximum treshold.

        masked : string
            Defines if the index is masked or not. Defaults to False.

        '''

        print('Tresholding...')

        if masked:
            index_p = self._get_image_path('masked', index)
        else:
            index_p = self._get_image_path('veg_index', index)

        with rasterio.open(index_p) as src:
            v_index = src.read(1)
            meta = src.meta

        if index == 'NDTI':

            v_index_tresh = np.where(((v_index < 1) & ((v_index <= tresh_value[0]) | (
                v_index >= tresh_value[1]))), 4, 0).astype('uint8')
            other_name = 'NDTI_other_2.tif'
            class_name = 'Soil_NDTI_tr.tif'

        elif index == 'BAEI':
            if self.level2A:
                with rasterio.open(self.scene_classification) as scl_src:
                    scl = scl_src.read(1)
            else:
                with rasterio.open(self.image) as scl_src:
                    scl = scl_src.read(7).astype('uint16')

            unclassified = np.where(scl == 7, 1, 0).astype('uint8')
            v_index_tresh = np.where(
                ((v_index >= tresh_value[0]) & (v_index <= tresh_value[1])), 5, 0)
            v_index_tresh = np.where((v_index_tresh - unclassified) == 5, 5, 0)

            mask_index_p = self._get_image_path('masked', 'NDTI_other_2')

            other_name = 'NDTI_other_3.tif'
            class_name = 'Built_up_BAEI_tr.tif'

            with rasterio.open(mask_index_p) as src:
                mask_index = src.read(1)
            mask = np.where(mask_index > 1, 0, 1)
            v_index_tresh = (v_index_tresh * mask).astype('uint8')
            v_index = mask_index

        v_index_tresh_inv = np.where(v_index_tresh == 0, 1, 0)
        v_index_mask = np.where(
            v_index * v_index_tresh_inv == 0, 99999, v_index).astype('float32')

        meta.update(
            nodata=99999
        )
        with rasterio.open(os.path.join(self._create_folder('masked'), other_name), 'w', **meta) as dst:
            if len(v_index_mask.shape) == 3:
                dst.write(v_index_mask)
            else:
                dst.write(v_index_mask, 1)

        meta.update(
            nodata=0,
            dtype='uint8'
        )

        with rasterio.open(os.path.join(self._create_folder('classes'), class_name), 'w', **meta) as dst:
            if len(v_index_tresh.shape) == 3:
                dst.write(v_index_tresh)
            else:
                dst.write(v_index_tresh, 1)

    def combine_calssifications(self, path):
        '''

        Combines all classifications into one raster:

        Parameters
        ----------

        path : string
            Folder where to save final classfication raster. If not provided, saves raster in current folder.

        '''

        print('Combining images...')

        for count, calssification in enumerate(os.listdir(self._create_folder('classes'))):
            if count == 0:
                with rasterio.open(os.path.join(self._create_folder('classes'), calssification)) as src:
                    final_class = src.read(1)
                    meta = src.profile
                continue
            with rasterio.open(os.path.join(self._create_folder('classes'), calssification)) as src:
                final_class = final_class + src.read(1)
        meta.update(
            tiled=True,
            blockxsize=256,
            blockysize=256
        )
        if not path:

            with rasterio.open(os.path.join(self._get_root(), 'ALCC.tif'), 'w', **meta) as dst:
                dst.write(final_class, 1)

        else:
            with rasterio.open(os.path.join(path, 'ALCC.tif'), 'w', **meta) as dst:
                dst.write(final_class, 1)

        print("Done!")

    def run(self, path=None):
        '''Runs entire ALCC algorithm.

        Parameters
        ---------
        path : string
            Folder where to save final classfication raster. If not provided, saves raster in current folder.
        '''

        if self.level2A:
            self.set_file_paths()
            self.resample_image(names=['SWIR1_10m', 'SWIR2_10m', 'SCL_10m'])

        self.calculate_vi(['MNDWI', 'NDVI', 'NDTI', 'BAEI'])
        self.mask_clouds()
        self.k_means_classification('MNDWI', k=3, masked=True)
        self.masking_image('NDVI', 'MNDWI')
        self.k_means_classification('NDVI', k=4, masked=True)
        self.masking_image('NDTI', 'NDVI')
        self.tresholding_classification(
            'NDTI', tresh_value=[0, 0.12], masked=True)
        self.tresholding_classification('BAEI', tresh_value=[0.45, 0.55])
        self.k_means_classification('NDTI_other_3', k=4, masked=True)
        self.extract_classes('NDTI', 'NDTI_other_3',
                             'NDTI_other_3_K4', path=True, masked=True)
        self.combine_calssifications(path)


class XGB_classification():
    def __init__(self, path_to_image):
        self.path = path_to_image

    @staticmethod
    def _get_root():
        return os.path.abspath(os.path.dirname(__file__))

    def _classify(self, stacked_raster, meta, name, path):

        row = stacked_raster.shape[0]
        col = stacked_raster.shape[1]
        blue = stacked_raster[:, :, 0]
        green = stacked_raster[:, :, 1]
        red = stacked_raster[:, :, 2]
        nir = stacked_raster[:, :, 3]
        swir1 = stacked_raster[:, :, 4]
        swir2 = stacked_raster[:, :, 5]
        scl = stacked_raster[:, :, 6]
        clouds = np.where((scl.ravel() == 8) | (scl.ravel() == 9))[0]
        ice = np.where(scl.ravel() == 11)[0]

        print('Collecting training data')
        # Empty raster
        supervised = np.full(shape=(row * col), fill_value=99, dtype=np.int)

        clouds_train = clouds[:int(clouds.shape[0]*0.2)]
        supervised[clouds_train] = 6

        ice_train = ice[:int(ice.shape[0]*0.2)]
        supervised[ice_train] = 8

        nodata = np.where(blue.ravel() == 0)[0]
        nodata_train = nodata[:int(nodata.shape[0]*0.01)]

        supervised[nodata_train] = 9

        # Water
        v_index = ((green - swir1) / (green + swir1)).astype('float32')
        v_index = np.nan_to_num(v_index)
        v_index = np.where(((v_index > 1) | (v_index < -1)), 0, v_index)

        water = np.where((v_index.ravel() > 0.4) & (supervised != 6) & (
            supervised != 8) & (supervised != 10))[0]

        water_train = water[:int(water.shape[0]*0.2)]

        supervised[water_train] = 1

        # High veg
        v_index = ((nir - red) / (nir + red)).astype('float32')
        v_index = np.nan_to_num(v_index)
        v_index = np.where(((v_index > 1) | (v_index < -1)), 0, v_index)

        high_veg = np.where((v_index.ravel() >= 0.6) & (supervised != 1) & (
            supervised != 6) & (supervised != 8) & (supervised != 10))[0]
        high_veg_train = high_veg[:int(high_veg.shape[0]*0.2)]
        supervised[high_veg_train] = 3

        # Low veg
        low_veg = np.where((v_index.ravel() < 0.6) & (v_index.ravel() >= 0.3) & (
            supervised != 1) & (supervised != 6) & (supervised != 8) & (supervised != 10))[0]
        lov_veg_train = low_veg[:int(low_veg.shape[0]*0.2)]
        supervised[lov_veg_train] = 2

        if high_veg_train[-1] <= lov_veg_train[-1]:
            lowest_index = int(high_veg_train[-1])
        else:
            lowest_index = int(lov_veg_train[-1])

        # Bare soil
        v_index = ((swir1 - swir2) / (swir1 + swir2)).astype('float32')
        v_index = np.nan_to_num(v_index)
        v_index = np.where(((v_index > 1) | (v_index < -1)), 0, v_index)

        bare_soil = np.where(((v_index.ravel() < 0) | (v_index.ravel() >= 0.12)) & (supervised != 1) & (
            supervised != 2) & (supervised != 3) & (supervised != 6) & (supervised != 8) & (supervised != 10))[0]
        bare_soil_train = bare_soil[:int(bare_soil.shape[0]*0.2)]
        bare_soil_train_lowest_index = bare_soil_train[bare_soil_train < lowest_index]

        supervised[bare_soil_train_lowest_index] = 4

        # Built up
        v_index = (red + 0.3) / (green + swir1).astype('float32')
        v_index = np.nan_to_num(v_index)
        v_index = np.where(((v_index > 1) | (v_index < -1)), 0, v_index)

        built_up = np.where(((v_index.ravel() >= 0.45) & (v_index.ravel() <= 0.55)) & (supervised != 1) & (supervised != 2) & (
            supervised != 3) & (supervised != 4) & (supervised != 6) & (supervised != 8) & (supervised != 10))[0]
        built_up_train = built_up[:int(built_up.shape[0]*0.2)]
        built_up_train_lowest_index = built_up_train[built_up_train < lowest_index]

        supervised[built_up_train_lowest_index] = 5

        y = supervised
        stacked_raster_without_scl = stacked_raster[:, :, :-1]

        X = stacked_raster_without_scl.reshape(
            (-1, stacked_raster_without_scl.shape[2])).astype('uint16')

        train = np.flatnonzero(supervised != 99)

        print('Classifiying...')
        data_matrix = xgb.DMatrix(data=X[train], label=y[train])
        params = {'objective': 'multi:softmax', 'num_class': 10,
                  'max_depth': 5, 'tree_method': 'exact'}
        xg_cls = xgb.train(params, data_matrix, num_boost_round=8)
        pred = xg_cls.predict(xgb.DMatrix(X))

        classification = pred.reshape(row, col)

        meta.update(
            count=1,
            nodata=9,
            dtype='uint8'
        )

        if not path:
            with rasterio.open(os.path.join(self._get_root(), '{}.tif'.format(name)), 'w', **meta) as dst:
                dst.write(classification.astype('uint8'), 1)
            return os.path.join(self._get_root(), '{}.tif'.format(name))

        else:
            with rasterio.open(os.path.join(path, '{}.tif'.format(name)), 'w', **meta) as dst:
                dst.write(classification.astype('uint8'), 1)
            return os.path.join(path, '{}.tif'.format(name))

    def _combine_split_classification(self, images, src_meta):
        src_files_to_mosaic = []
        src_1 = rasterio.open(images[0])
        src_files_to_mosaic.append(src_1)
        src_2 = rasterio.open(images[1])
        src_files_to_mosaic.append(src_2)
        mosaic, out_transform = merge(src_files_to_mosaic)
        out_meta = src_meta
        out_meta.update({"driver": "GTiff",
                         "height": mosaic.shape[1],
                         "width": mosaic.shape[2],
                         "transform": out_transform,
                         "tiled": True,
                         "count":1,
                         "dtype":"uint8",
                         "nodata":9,
                         }
                        )
        mosaic_file = str(images[0]).replace('_part_1', '') + '.tif'
        with rasterio.open(mosaic_file, "w", **out_meta) as dest:
            dest.write(mosaic)
        src_1.close()
        src_2.close()
        for image in images:
            os.remove(image)

    def run(self, path=None, name=None, split=False):

        if name is None:
            name = 'xgb_classification'
        else:
            try:
                name = str(name)
            except Exception as e:
                print(e)
            else:
                name = 'xgb_classification_{}'.format(name)

        with rasterio.open(self.path) as src:
            stacked_raster = reshape_as_image(src.read().astype('float32'))
            meta = src.meta.copy()
        if split:
            src_transform = src.transform
            half_height = int(src.height/2)
            image_paths = []
            for i in range(2):
                name_split = name + '_part_{}'.format(i+1)
                if path:
                    if os.path.exists(os.path.join(path, name_split + '.tif')):
                        image_paths.append(os.path.join(path, name_split + '.tif'))
                        continue
                if i == 0:
                    stacked_raster_split = stacked_raster[:half_height, :]
                    meta.update(
                        height=half_height,
                    )
                    print('Classifiying first half...')
                    image_paths.append(self._classify(
                        stacked_raster_split, meta, name_split, path))
                else:
                    stacked_raster_split = stacked_raster[half_height:, :]
                    dst_transform = src_transform*A.translation(
                        0, half_height)
                    meta.update(
                        transform=dst_transform
                    )
                    print('Classifiying second half...')
                    image_paths.append(self._classify(
                        stacked_raster_split, meta, name_split, path))

            self._combine_split_classification(image_paths, src.meta.copy())
        else:
            self._classify(stacked_raster, meta, name, path)
        print('Done!')
