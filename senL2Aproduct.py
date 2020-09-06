# Standard library
import sys
from io import BytesIO
import math
import os
from zipfile import ZipFile
import re

# General imports
import numpy as np
import matplotlib.pyplot as plt
import requests
import pandas
from bs4 import BeautifulSoup
from PIL import Image
from clint.textui import progress

# Spatial imports
import pyproj
import shapely
import fiona
import geopandas
import rasterio
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.io.img_tiles as cimgt


class SenL2Aproduct:

    '''Class to search and download multispectral satellite images aquired by ESA Sentinel 2 satellites.
    Product (image) is Level-2A product which represents Bottom-Of-Atmosphere (BOA) reflectance.
    Each product is 100 x 100 km2 ortho-image in UTM/WGS84 projection.

    To use the api you must create a free account on Copernicus Open Acces Hub
    https://scihub.copernicus.eu/dhus/#/self-registration

    Parameters
    ----------
    username : string
        username for Copernicus account
    password : string 
        password for Copernicus account
    area_of_interest : string, object
        area used to filter search for satellite images. Only those images that intersect area of interest are chosen. Supported formats are:
            - WKT
            - shapefile
            - shapley Polygon or Multypolygon
            - geojson
            - name of a county or country
            *only Croatia and Croatian counties are available. To see available countries/counties use method "print_countries/print_counties".
        If there is no information on area crs it is assumed it's in WGS84.
    period : string
        time period in which to search satellite images. For example:
        [2019-01-01 TO NOW], [2019-01-01 TO 2019-02-01], [NOW-2DAYS TO NOW]
        for more examples visit https://scihub.copernicus.eu/userguide/FullTextSearch 
        *no need to specifiy hh:mm:ss, default T00:00:00.000Z is appended for each yyyy-MM-dd date.' Defaults to last 30 days (NOW-30DAYS TO NOW)
    cloudcover : string
        precentage of cloud cover in satellite images. Defaults to a range [0 TO 30]


    Attributes
    ---------
    counry : dict
        Counries to use as area of interest. To see available countries use method print_countries.
        *currently only Croatia is avaliable.

    counties : dict
        Counties to use as area of interest. To see available counties use method print_counties.
        *currently only Croatian counties are avaliable.

    src_crs : bool
        Area of interest crs. Defaults to None.

    '''

    countries = {
        'hrvatska': 'HR0'
    }

    counties = {
        'zagrebacka': 'HR065',
        'krapinsko-zagorska': 'HR064',
        'koprivnicko-krizevacka': 'HR063',
        'varazdinska': 'HR062',
        'medimurska': 'HR061',
        'grad_zagreb': 'HR050',
        'dubrovacko-neretvanska': 'HR037',
        'istarska': 'HR036',
        'splitsko-dalmatinska': 'HR035',
        'sibensko-kninska': 'HR034',
        'zadarska': 'HR033',
        'licko-senjska': 'HR032',
        'primorsko-goranska': 'HR031',
        'sisacko-moslavacka': 'HR028',
        'karlovacka': 'HR027',
        'vukovarsko-srijemska': 'HR026',
        'osjecko-baranjska': 'HR025',
        'brodsko-posavska': 'HR024',
        'pozesko-slavonska': 'HR023',
        'viroviticko-podravska': 'HR022',
        'bjelovarsko-bilogorska': 'HR021'
    }

    src_crs = False

    def __init__(self, username, password, area_of_interest=None, period='[NOW-30DAYS TO NOW]', cloudcover='[0 TO 30]'):
        self.credentials = (username, password)
        self.period = SenL2Aproduct._check_period(period)
        self.area_of_interest = SenL2Aproduct._format_area(area_of_interest)
        self.cloudcover = cloudcover
        self.target_crs = 3857

    @staticmethod
    def _sort_gdf(gdf, columns, ascending=True):
        if not isinstance(columns, list):
            columns = [columns]
        return gdf.sort_values(columns, ascending=ascending)

    @classmethod
    def print_counties(cls):
        print([county for county in cls.counties.keys()])

    @classmethod
    def print_countries(cls):
        print([country for country in cls.countries.keys()])

    @staticmethod
    def _get_temp_path():
        root = os.path.abspath(os.path.dirname(__file__))
        temp_path = os.path.join(root, 'tmp')
        if not os.path.exists(temp_path):
            os.mkdir(temp_path)
        return temp_path

    def __repr__(self):
        return "SenL2Aproduct({}, {}, {}, {}, {})".format(self.credentials[0], self.credentials[1], self.area_of_interest, self.period, self.cloudcover)

    def get_period(self):
        return self.period

    def set_period(self, period):
        self.period = SenL2Aproduct._check_period(period)

    def get_cloudcover(self):
        return self.cloudcover

    def set_cloudcover(self, cloudcover):
        self.cloudcover = cloudcover

    def get_src_crs(self):
        return self.src_crs

    @classmethod
    def set_src_crs(cls, src_crs, gp=False):
        if gp:
            src_crs = int(src_crs['init'].split(':')[1])
        elif isinstance(src_crs, str):
            src_crs = int(src_crs)
        cls.src_crs = src_crs

    def get_area_of_interest(self):
        return self.area_of_interest

    def set_area_of_interest(self, aoi):
        self.area_of_interest = SenL2Aproduct._format_area(aoi)

    @classmethod
    def _format_area(cls, area):

        if not area:
            return None

        if isinstance(area, shapely.geometry.Polygon) or isinstance(area, shapely.geometry.MultiPolygon):
            cls.set_src_crs(4326)
            return area

        elif '.shp' in area or '.geojson' in area:
            try:
                aoi = geopandas.read_file(area)
            except fiona.errors.DriverError:
                print('No file named {}'.format(area))
                sys.exit(0)
            else:
                cls.set_src_crs(aoi.crs, gp=True)
                return aoi.geometry[0]

        elif area in [key for key in cls.counties.keys()]:
            aoi = geopandas.read_file(
                'https://gisco-services.ec.europa.eu/distribution/v2/nuts/distribution/{}-region-01m-3857-2021.geojson'
                .format(cls.counties[area])
            )
            cls.set_src_crs(3857)
            return aoi.geometry[0]

        elif area in [key for key in cls.countries.keys()]:
            aoi = geopandas.read_file(
                'https://gisco-services.ec.europa.eu/distribution/v2/nuts/distribution/{}-region-01m-3857-2021.geojson'
                .format(cls.countries[area])
            )
            cls.set_src_crs(3857)
            return aoi.geometry[0]

        else:
            try:
                aoi = geopandas.read_file(area)
            except ValueError:
                try:
                    aoi = shapely.wkt.loads(area)
                except Exception as e:
                    print('{} \n'
                          'Area is not a county or country. For avaliable counties/country run SenL2Aproduct.print_counties()/.print_countries(). \n Area is not in valid WKT or geojson format.'.format(e))
                    sys.exit(0)
                else:
                    if not cls.src_crs:
                        cls.set_src_crs(4326)
                    return aoi
            else:
                cls.set_src_crs(aoi.crs, gp=True)
                return aoi.geometry[0]

    @ classmethod
    def _check_period(cls, period):

        try:
            if not isinstance(period, str):
                raise TypeError

        except TypeError:
            print('Date must be provided as a string')

        else:
            if not 'TO' in period.upper():
                raise ValueError('Wrong date format. Correct examples: \n\
                    [2019-01-01 TO NOW], \n \t\t[2019-01-01 TO 2019-02-01], \n \t\t[NOW-2DAYS TO NOW] \n')

            if '[' in period:
                period = period.strip()[1:-1]

            begin_date = period.upper().split('TO')[0].replace(" ", "")
            end_date = period.upper().split('TO')[1].replace(" ", "")
            hms = 'T00:00:00.000Z'

            if 'NOW' in begin_date and 'NOW' in end_date:
                pass

            elif 'NOW' in begin_date:
                end_date = end_date + hms

            elif 'NOW' in end_date:
                begin_date = begin_date + hms
            else:

                begin_y_m = begin_date.split('-')[:-1]
                if int(begin_y_m[0]) < 2018:
                    raise ValueError('No Level-2A products before March 2018.')

                elif int(begin_y_m[0]) == 2018 and int(begin_y_m[1]) < 3:
                    raise ValueError('No Level-2A products before March 2018.')

                begin_date = begin_date + hms
                end_date = end_date + hms

            return '[{} TO {}]'.format(begin_date, end_date)

    def _transform_crs(self, src, target, geom):
        src_crs = pyproj.CRS.from_epsg(src)
        target_crs = pyproj.CRS.from_epsg(target)

        project = pyproj.Transformer.from_crs(
            src_crs, target_crs, always_xy=True).transform
        target = shapely.ops.transform(project, geom)
        target_rounded = shapely.wkt.loads(
            shapely.wkt.dumps(target, rounding_precision=4))

        return target_rounded

    @staticmethod
    def _create_gdf():
        return geopandas.GeoDataFrame(pandas.DataFrame(columns=[
            'Title', 'Date', 'Clouds', 'Size', 'ID', 'Footprint', 'Intersection_area', 'Intersection_geom']), geometry='Footprint')

    def _create_search_url(self):
        if self.src_crs != 4326:
            search_bbox = self._transform_crs(
                self.src_crs, 4326, self.area_of_interest).convex_hull
        else:
            search_bbox = self.area_of_interest.convex_hull
        search_url = 'https://scihub.copernicus.eu/dhus/search?start=0&rows=100&q=platformname:Sentinel-2 AND producttype:S2MSI2A AND footprint:"Intersects({})" AND cloudcoverpercentage:{} AND beginposition:{}'.format(
            search_bbox, self.cloudcover, self.period)
        return search_url

    def _spatial_op(self, op, geom_1, geom_2, area=False):
        if op == 'intersection':
            sp_op = geom_1.intersection(geom_2)
        elif op == 'difference':
            sp_op = geom_1.difference(geom_2)
        elif op == 'union':
            sp_op = geom_1.union(geom_2)

        sp_op_rounded = shapely.wkt.loads(
            shapely.wkt.dumps(sp_op, rounding_precision=4))
        sp_op_valid = self._check_geom(sp_op_rounded)

        if area:
            return (sp_op_valid, sp_op.area)
        return sp_op_valid

    def _check_geom(self, geom):
        try:
            if not geom.is_valid:
                geom = geom.buffer(0)
                if not geom.is_valid:
                    raise shapely.errors.TopologicalError
            return geom
        except shapely.errors.TopologicalError:
            print('Invalid geometry.')
            return None

    def _get_product_properties(self, entry):

        title = entry.title.get_text()
        date = entry.date.get_text()
        clouds = float(entry.find(
            'double', {'name': 'cloudcoverpercentage'}).text)
        size = entry.find('str', {'name': 'size'}).text
        if 'MB' in size.upper():
            size = float(size.split(' ')[0])/1000
        else:
            size = float(size.split(' ')[0])
        ID = entry.id.get_text()
        footprint = shapely.wkt.loads(
            entry.find('str', {'name': 'footprint'}).text)

        return (title, date, clouds, size, ID, footprint)

    def _make_request(self, search_url, bs=False):
        req = requests.get(url=search_url, auth=self.credentials)
        if bs:
            return BeautifulSoup(req.text, 'html.parser'), req.status_code
        return req, req.status_code

    def _count_products(self, products):

        products_found = products.subtitle.text.split()

        try:
            # Found more than 100 products
            if 'to' in products_found:
                print('Found total {} products.'.format(products_found[5]))
                print('Working with first 100. To decrease number of products found try decreasing period, cloud cover percentage or area of interest.')
                num_of_products = int(products_found[3])
            else:
                num_of_products = int(products_found[1])
                print('Found {} products.'.format(num_of_products))
        except:
            print('No products found')
            sys.exit(0)

        return num_of_products

    def search_products(self, cover_aoi=False, same_date=False):
        '''Returns a Geodataframe of products found.

        Parameters
        ----------
        cover_aoi : bool
            Checks if each image completely covers area of interest.
            If it does, only that image is returned.
            Else, returns images that, when combined, cover area of interest. Defaults to False.
        same_date : bool
            Only aplicable when cover_aoi is True.
            Checks that images that cover aoi have the same date.
            If there are multiple groups of images that cover aoi, returns group with the smallest collective cloud cover. Defaults to False
        '''

        if not self.area_of_interest:
            print('To search products you must provide area of interest!')
            return

        print("Searching images...")

        print(self._create_search_url())

        products_req, products_status_code = self._make_request(self._create_search_url(), bs=True)

        if products_status_code != 200:
            print('Invalid request. Possible typo in username and password!?')
            sys.exit(0)

        num_of_products = self._count_products(products_req)

        products_gdf = SenL2Aproduct._create_gdf()

        if num_of_products > 0:

            if self.src_crs != self.target_crs:
                self.area_of_interest = self._transform_crs(
                    self.src_crs, self.target_crs, self.area_of_interest)

            for entry in products_req.find_all('entry'):

                title, date, clouds, size, ID, footprint = self._get_product_properties(
                    entry)

                footprint = self._transform_crs(
                    4326, self.target_crs, footprint)

                # Check if product really intersects aoi
                if footprint.intersects(self.area_of_interest):

                    intersection_geom, intersection_area = self._spatial_op('intersection',
                                                                            footprint, self.area_of_interest, area=True)

                    products_gdf = products_gdf.append(
                        {'Title': title, 'Date': date, 'Clouds': clouds, 'Size': size, 'ID': ID, 'Footprint': footprint, 'Intersection_area': intersection_area/1000000, 'Intersection_geom': intersection_geom.wkt}, ignore_index=True)

            products_gdf.crs = {'init': self.target_crs}

            if cover_aoi:
                self.products = self._select_aoi_products(
                    products_gdf, same_date)

            else:
                self.products = products_gdf
        else:
            print('Try expanding search period and cloud cover percentage.')
            return products_gdf

    def _check_products_overlap(self, current_geom, union_geom):

        if union_geom.buffer(1).contains(current_geom):
            return None

        elif union_geom.intersects(current_geom):
            overlap_geom = self._spatial_op(
                'intersection', union_geom, current_geom)
            difference_geom = self._spatial_op(
                'difference', current_geom, overlap_geom)
            return difference_geom

        else:
            return current_geom

    def _check_union_contains_aoi(self, union, tolerance=5):
        if int(union.area) >= int(self.area_of_interest.area)-tolerance:
            return True
        return False

    def _select_best_date(self, current_clouds, best_clouds, current_date):
        if best_clouds is None or (best_clouds > current_clouds):
            #print('Best date is {}'.format(current_date))
            return current_clouds, current_date
        else:
            #print("Previous clouds {} < current clouds {}".format(
                best_clouds, current_clouds))
            #print(current_date)
            return current_date

    # Selects products that cover area of interest.
    def _select_aoi_products(self, products, same_date):

        previous_product = None
        products_found = False
        multiple_img = True
        products_list = []

        if same_date:
            products_sorted = SenL2Aproduct._sort_gdf(
                products, ['Date', 'Intersection_area'], ascending=False)
        else:
            products_sorted = SenL2Aproduct._sort_gdf(
                products, 'Intersection_area', ascending=False)

        for _, product in products_sorted.iterrows():
            current_date = product.Date
            current_clouds = product.Clouds
            current_geom = self._check_geom(
                shapely.wkt.loads(product.Intersection_geom))

            # Check if one product covers aoi
            if self._check_union_contains_aoi(current_geom):
                #print('Single geom covers aoi')
                multiple_img = False
                products_found = True
                products_list = [product.Title]
                if same_date:
                    check_date = self._select_best_date(
                        current_clouds, best_clouds, current_date)
                    if isinstance(check_date, tuple):
                        best_clouds, best_date = check_date
                        best_products = [product for product in products_list]
                    else:
                        second_best_date = check_date
                else:
                    break
            if multiple_img:
                if len(products_list) == 0:
                    union_geom = current_geom
                    products_list = [product.Title]
                    if same_date:
                        previous_product = product
                        clouds_sum = current_clouds
                        best_date = None
                        second_best_date = None
                        best_clouds = None
                    continue

                if same_date:
                    if previous_product.Date != current_date:
                        union_geom = current_geom
                        products_list = [product.Title]
                        previous_product = product
                        clouds_sum = current_clouds
                        continue
                    elif current_date == best_date or current_date == second_best_date:
                        #print('Already checked date {}'.format(current_date))
                        continue

                # Calculate difference between current product and previous products (union)
                difference_geom = self._check_products_overlap(
                    current_geom, union_geom)
                if difference_geom:
                    union_geom = self._spatial_op(
                        'union', union_geom, difference_geom)
                    products_list.append(product.Title)
                    if same_date:
                        clouds_sum += current_clouds
                    if self._check_union_contains_aoi(union_geom):
                        products_found = True
                        if same_date:
                            check_date = self._select_best_date(
                                clouds_sum, best_clouds, current_date)
                            if isinstance(check_date, tuple):
                                best_clouds, best_date = check_date
                                best_products = [
                                    product for product in products_list]
                            else:
                                second_best_date = check_date
                        else:
                            break

        if products_found:
            if same_date:
                if len(best_products) == 1:
                    print('Found {} product that covers AOI.'.format(
                        len(best_products)))
                else:
                    print('Found {} products with same date that cover AOI.'.format(
                        len(best_products)))
                products_found = products_sorted[products_sorted.Title.isin(
                    best_products)]
                return SenL2Aproduct._sort_gdf(products_found, 'Title', ascending=False)
            else:
                if len(products_list) == 1:
                    print('{} product covers AOI'.format(len(products_list)))
                else:
                    print('{} products cover AOI'.format(len(products_list)))
                products_found = products_sorted[products_sorted.Title.isin(
                    products_list)]
                return SenL2Aproduct._sort_gdf(products_found, 'Title', ascending=False)
        else:
            print("Didn't find any products that completely cover AOI. Found {} products that cover {} % of the AOI. Try expanding search period and/or increasing cloud cover percantange range.".format(
                len(products_list), np.round((union_geom.area / self.area_of_interest.area), decimals=2)*100))
            products_found = products_sorted[products_sorted.Title.isin(
                products_list)]
            return SenL2Aproduct._sort_gdf(products_found, 'Title', ascending=False)

    def _get_preview(self, prod_id, prod_title, ext='.jpg'):
        img_path = os.path.join(SenL2Aproduct._get_temp_path(), prod_title + ext)
        if os.path.exists(img_path):
            print('{} preview in cache'.format(prod_title))
            with Image.open(img_path) as img:
                return np.asarray(img)
        print('Downloading preview for product:{}'.format(prod_title))
        preview_url = "https://scihub.copernicus.eu/dhus/odata/v1/Products('{}')/Products('Quicklook')/$value".format(
            prod_id)
        preview_img_byte = self._download_preview(preview_url)
        preview_img_arr = SenL2Aproduct._cache_preview(
            preview_img_byte, img_path)
        return preview_img_arr

    def _download_preview(self, url):
        previews_req = requests.get(url, auth=self.credentials)
        return previews_req.content

    @staticmethod
    def _cache_preview(img_byte, name):
        print('Caching preview...')
        with Image.open(BytesIO(img_byte)) as img:
            img.save(name, format='jpeg')
            return img

    @staticmethod
    def _get_nrow_ncol(num_of_prod):
        if num_of_prod >= 5:
            return 3, 4
        elif num_of_prod >= 3:
            return 2, 4
        elif num_of_prod == 2:
            return 1, 4
        else:
            return 1, 2

    @staticmethod
    def _set_nrow_ncol(products, page):
        products_len = len(products)
        full_pages = math.floor(products_len/6)
        if full_pages == 0:
            full_pages = all_pages = 1
            num_products = products_len
        else:
            num_products = 6
            if (products_len/6).is_integer():
                all_pages = full_pages
            else:
                all_pages = full_pages + 1
        try:
            if page > all_pages:
                raise IndexError
            elif page <= full_pages:
                nrow, ncol = SenL2Aproduct._get_nrow_ncol(num_products)
            else:
                num_products = products_len - (6 * full_pages)
                nrow, ncol = SenL2Aproduct._get_nrow_ncol(num_products)
        except IndexError:
            print('No more images to show')
            print('Last page is {}'.format(all_pages))
            sys.exit(0)

        return nrow, ncol, num_products

    @staticmethod
    def _get_polygon_extent(polygon):

        lonlist, latlist = polygon.envelope.exterior.xy

        lat_min = np.min(latlist)
        lon_min = np.min(lonlist)
        lat_max = np.max(latlist)
        lon_max = np.max(lonlist)

        return [lon_min, lon_max, lat_min, lat_max]

    def _plot_aoi_foot(self, ax, footprint, extent):
        ax.set_extent(extent, crs=ccrs.epsg(3857))
        ax.stock_img()
        #ax_geom.add_image(request, 10)
        ax.add_geometries(footprint, ccrs.epsg(3857), alpha=0.4, color='r')
        if self.area_of_interest.type == 'Polygon':
            aoi_mp = shapely.geometry.MultiPolygon(
                polygons=[self.area_of_interest])
            ax.add_geometries(aoi_mp, ccrs.epsg(3857), alpha=0.4, color='b')
        else:
            ax.add_geometries(self.area_of_interest,
                              ccrs.epsg(3857), alpha=0.4, color='b')
        gl = ax.gridlines(draw_labels=True, alpha=0.2)
        gl.xformatter = LONGITUDE_FORMATTER
        gl.xlabel_style = {'size': 9}
        gl.top_labels = False
        gl.xlines = False
        gl.yformatter = LATITUDE_FORMATTER
        gl.ylabel_style = {'size': 9}
        gl.right_labels = False
        gl.ylines = False

    def show_products(self, page=1):
        '''Shows Sentinel level 2A products (RGB satellite image) alongside their footprint and area of interest polygon. If there are more than 6 products, shows first 6. For next products increment page parameter.
        Preview images are cached in directory "tmp".

        Parameters:
        ----------
        page : int
            Selects which page to show. Pages start at 1. Defaults to 1.
        '''

        if not 'products' in self.__dir__():
            print('No products have been found, must first search.')
            return

        if page == 0:
            page = 1

        nrow, ncol, num_products = self._set_nrow_ncol(self.products, page)

        start_index = (page - 1) * 6

        fig = plt.figure(figsize=(13, 10))

        #request = cimgt.StamenTerrain()

        for i in range(1, num_products + 1):
            product = self.products.iloc[start_index + i - 1]
            product_index = self.products.index.values.tolist()[i-1]
            footprint = product.Footprint
            foot_aoi_union = footprint.union(
                self.area_of_interest).buffer(10000)
            extent = SenL2Aproduct._get_polygon_extent(foot_aoi_union)

            ax_geom = fig.add_subplot(
                nrow, ncol, i*2-1, projection=ccrs.epsg(3857))
            ax_geom.set_title(product.Title, fontsize=7)
            self._plot_aoi_foot(ax_geom, footprint, extent)

            ax_img = fig.add_subplot(nrow, ncol, i*2)
            ax_img.set_title(product_index, fontsize=7)
            preview_img = self._get_preview(product.ID, product.Title)
            ax_img.imshow(preview_img)
            ax_img.axis('off')

            plt.tight_layout(pad=1.2)

        plt.show()

    def download_products(self, products=None, all=False, extract=False):
        '''Downloads Sentinel2 L2A product(s) in a directory called "products".

        Parameters:
        ----------

        products : list
            List of product indexes for products you want to download. Index of a product is displayed above satelite image when previewing products. Defaults to None.
        all : bool
            If True downloads all products found. Deaults to False.
        extract : bool
            If True extracts (unzips) file and deletes original zip. Defaults to False. 

        '''
        if not products and not all:
            print('No products have been found, must first search.')
            return 

        try:
            if 'products' in self.__dir__():
                SenL2Aproduct._create_download_dir()
                for index, product in self.products.iterrows():
                    dl_url = self._create_download_url(product.ID)
                    product_req, _ = self._make_request(dl_url)
                    file_name = product.Title + '.zip' 
                    if all:
                        if os.path.exists(file_name) or os.path.exists(file_name.split('.')[0] + '.SAFE'):
                            print('Already downloaded {}. Moving on to the next product'.format(file_name.split('.')[0]))
                            continue
                        self._chunk_download(product_req, file_name, extract)
                    else:
                        if index in products:
                            if os.path.exists(file_name) or os.path.exists(file_name.split('.')[0] + '.SAFE'):
                                print('Already downloaded {}. Moving on to the next product'.format(file_name.split('.')[0]))
                                continue
                            print('Downloading index {}'.format(index))
                            self._chunk_download(
                                product_req, file_name, extract)
            else:
                raise AttributeError
        except AttributeError:
            print('No products have been found, must first search.')
            sys.exit(0)
        os.chdir(os.path.abspath(os.path.dirname(__file__)))

    def _create_download_url(self, prod_id):
        download_url = "https://scihub.copernicus.eu/dhus/odata/v1/Products('{}')/$value".format(
            prod_id)
        return download_url

    def _chunk_download(self, req, file_name, extract):
        print('Downloading product {}...'.format(file_name))
        with open(file_name, "wb") as f:
            total_length = int(req.headers.get('content-length'))
            for chunk in progress.bar(req.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1):
                if chunk:
                    f.write(chunk)
                    f.flush()
        if extract:
            self._extract_zip(file_name)

    @staticmethod
    def _create_download_dir():
        root = os.path.abspath(os.path.dirname(__file__))
        download_dir = os.path.join(root, 'products')
        if not os.path.exists(download_dir):
            os.mkdir(download_dir)
        os.chdir(download_dir)

    def _extract_zip(self, zipPath):
        with ZipFile(zipPath, 'r') as zipimg:
            zipimg.extractall()
        os.remove(zipPath)




def _create_folder(name):
    # Creates folder in current directory
    root = os.path.abspath(os.path.dirname(__file__))
    new_dir = os.path.join(root, name)
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    return new_dir


def _find_band(images_path, bands=None, resolution=None):
    if resolution:
        for dirpath, dirnames, files in os.walk(images_path):
            if dirpath.endswith(str(resolution).replace(' ', '')):
                if bands:
                    for image in files:
                        if '.xml' in image:
                            continue
                        for band in bands:
                            # Checking if band is one digit to prevent selecting two digit band with same first digit.
                            if len(str(band)) == 1:
                                if str(band) in image.split('_')[-2][-1] and image.split('_')[-2][-2] == '0':
                                    yield os.path.join(dirpath, image)
                            else:
                                if str(band) in image.split('_')[-2]:
                                    yield os.path.join(dirpath, image)
                else:
                    for image in files:
                        if '.xml' in image:
                            continue
                        yield os.path.join(dirpath, image)
    else:
        try:
            if not bands:
                raise ValueError
        except ValueError as e:
            print('If no resolution is provided, must provide array of bands.')
            print(e)
        else:
            for dirpath, dirnames, files in os.walk(images_path):
                if dirpath.endswith('10m') or dirpath.endswith('20m') or dirpath.endswith('60m'):
                    for image in files:
                        if '.xml' in image:
                            continue
                        for band in bands:
                            if len(str(band)) == 1:
                                if str(band) in image.split('_')[-2][-1] and image.split('_')[-2][-2] == '0':
                                    #print(os.path.join(dirpath, image))
                                    yield os.path.join(dirpath, image)
                            else:
                                if str(band) in image.split('_')[-2]:
                                    #print(os.path.join(dirpath, image))
                                    yield os.path.join(dirpath, image)


def _find_band_resolution(folder_path, bands, resolution):

    images_to_combine = []

    for path in _find_band(folder_path, bands, resolution):
        images_to_combine.append(path)
    if len(images_to_combine) < len(bands):
        if resolution == '10m':
            remaining_bands = [chn for chn in bands if chn not in [
                'AOT', 2, 3, 4, 8, 'TCI', 'WVP']]
            for path in _find_band(folder_path, remaining_bands, '20m'):
                images_to_combine.append(path)
            if len(images_to_combine) < len(bands):
                remaining_bands = [chn for chn in bands if chn in [1, 9]]
                for path in _find_band(folder_path, remaining_bands, '60m'):
                    images_to_combine.append(path)
        elif resolution == '20m':
            remaining_bands = [chn for chn in bands if chn in [1, 8, 9]]
            if 8 in remaining_bands:
                for path in _find_band(folder_path, [8], '10m'):
                    images_to_combine.append(path)
                remaining_bands.remove(8)
            if len(images_to_combine) < len(bands):
                for path in _find_band(folder_path, remaining_bands, '60m'):
                    images_to_combine.append(path)
        else:
            for path in _find_band(folder_path, [8], '10m'):
                images_to_combine.append(path)

    return images_to_combine


def clip_images(images, shape, shape_crs=3857, bands=None, resolution=None, extension='_clip', folder='clip'):
    '''Clips satellite images with vector geometry. 

    Parameters:
    ----------

    images : string
        Path to a Sentinel Level-2A product, or folder containing multiple Level-2A products (unzipped).
    shape : string, object
        Geometry used for clipping. Can be path to a file (shapefile or geojson) or a shapely object.
    shape_crs : int
        CRS of geometry used for clipping. Defaults to Web Mercator (3857).
    bands : object
        List containing band numbers to clip. Sentinel Level-2A has 13 bands plus additional products: AOT, SCL, TCI and WVP. 
    resolution : object
        List containing band resolution(s). Only bands with given resolution will be clipped. Possible options are: 10m, 20m, and 60m. If none is provided, bands of all resoution will be clipped. Defaults to None.
    extension : string
        Suffix to append to a clipped image. Defaults to "_clip".
    folder : string
        Path to a folder where clipped images will be saved. Defaults to folder "clip" in current directory.
    '''

    clip_dir = _create_folder(folder)

    if isinstance(shape, shapely.geometry.Polygon) or isinstance(shape, shapely.geometry.MultiPolygon):
        clip_geom = shape
    else:
        try:
            clip_geom = geopandas.read_file(shape).geometry[0]
        except Exception as e:
            print(e)

    if resolution:
        for res in resolution:
            for path in _find_band(images, bands, res):
                _clip_and_save(path, clip_geom, shape_crs, clip_dir, extension)
    else:
        for path in _find_band(images, bands):
            _clip_and_save(path, shape, shape_crs, clip_dir, extension)


def _clip_and_save(image, geom, geom_crs, dir_path, extension):

    clip_image_name = image.split('\\')[-1].split('.')[0] + extension + '.tif'

    if os.path.exists(os.path.join(dir_path, clip_image_name)):
        print("Already clipped {}".format(clip_image_name))
        return
    
    if geom.type == 'Polygon':
        geom = shapely.geometry.MultiPolygon(
            polygons=[geom])

    with rasterio.open(image) as src:
        raster_crs = int(str(src.crs).split(':')[1])
        if raster_crs != geom_crs:
            # print('Different crses')
            geom = _transform_crs(geom_crs, raster_crs, geom)
        try:
            out_image, out_transform = mask(src, geom, crop=True)
            out_meta = src.meta
            out_meta.update({"driver": "GTiff",
                            "height": out_image.shape[1],
                            "width": out_image.shape[2],
                            "transform": out_transform,
                            "nodata": 0
                            })
            with rasterio.open(os.path.join(dir_path, clip_image_name), "w", **out_meta, overwrite=True) as dest:
                print("Saving clipped image...")
                dest.write(out_image)
        except:
            pass


def _transform_crs(src, target, geom):
    src_crs = pyproj.CRS.from_epsg(src)
    target_crs = pyproj.CRS.from_epsg(target)
    project = pyproj.Transformer.from_crs(
        src_crs, target_crs, always_xy=True).transform
    target = shapely.ops.transform(project, geom)
    target_rounded = shapely.wkt.loads(
        shapely.wkt.dumps(target, rounding_precision=4))
    return target_rounded


def create_mosaic(bands, folder_path, resolution='10m', level2A=False):
    '''Creates image mosaic from satellite images and saves it in a new folder "mosaic".

    bands : object
        List of band numbers to mosaic. Mosaic is created for each band.
    folder_path : string
        Path to a folder containing satelite images. If folder contains level 2A products, then "level2A" must be set to True. If not, then satellite images must have band number in their name. Example: image1_02.tif.
    resolution : string
        Target resolution for image mosaic. Possible options are: 10m, 20m and 60m. Defaults to 10m.
    level2A : bool
        Determines whether the folder path is level 2A product. Defaults to False.

    Example:

        create_mosaic(bands=[2,3,4,8,11,12,'SCL'], folder_path='S2B_MSIL2A_20200207T095059_N0214_R079_T33TVJ_20200207T120904.SAFE', level2A=True)
    '''
    try:
        if str(resolution.replace(' ', '')) not in ['10m', '20m', '60m']:
            raise ValueError
    except ValueError:
        print('ValueError: Resoultion not supported. Possible options are: 10m, 20m or 60m')
        sys.exit(0)

    mosaic_dir = _create_folder('mosaic')

    for band in bands:
        images_to_mosaic = []
        if len(str(band)) == 1:
            band_full_name = ''.join(['B', '0', str(band)])
        elif len(str(band)) == 2:
            band_full_name = ''.join(['B', str(band)])
        else:
            band_full_name = band
        if level2A:
            images_to_mosaic = _find_band_resolution(
                folder_path, str(band), resolution)
        else:
            for image in os.listdir(folder_path):
                if band_full_name in image:
                    try:
                        with rasterio.open(os.path.join(folder_path, image)) as src:
                            images_to_mosaic.append(
                                os.path.join(folder_path, image))
                    except:
                        pass
            if len(images_to_mosaic) == 0:
                print('No images in band {}.'.format(band))
                continue
        _create_and_save_mosaic(
            images_to_mosaic, mosaic_dir, band_full_name, resolution)


def _create_and_save_mosaic(images, mosaic_dir, band, resolution):
    src_files_to_mosaic = []

    for i, img in enumerate(images):
        src = rasterio.open(img)
        if i == 0:
            dst_crs = src.crs
        if src.crs != dst_crs:
            print('Reprojecting raster...')
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds)
            meta = src.meta
            meta.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height
            })
            reprojected_img = img.split('.')[0] + '_reprojected.tif'
            with rasterio.open(reprojected_img, 'w', **meta) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest)
            src = rasterio.open(reprojected_img)

        if int(src.transform[0]) == int(resolution[:-1]):
            src_files_to_mosaic.insert(0, src)
        else:
            src_files_to_mosaic.append(src)

    mosaic, out_transform = merge(src_files_to_mosaic)
    out_meta = src.meta.copy()
    out_meta.update({"driver": "GTiff",
                     "height": mosaic.shape[1],
                     "width": mosaic.shape[2],
                     "transform": out_transform,
                     "tiled": True,
                     "crs": CRS.from_epsg(32633)
                     }
                    )
    mosaic_file = os.path.join(mosaic_dir, band)
    with rasterio.open('{}_mosaic.tif'.format(mosaic_file), "w", **out_meta) as dest:
        print('Creating mosaic for band {}...'.format(band))
        dest.write(mosaic)


def stack_images(bands, folder_path, resolution='10m', level2A=False):
    '''Stacks multiple satellite bands into one multiband image. New stacked image is saved in folder "stacked".

    bands : object
        List of band numbers to stack. Band number should be in image name.
    folder_path : string
        Path to a folder containing satelite images. If folder contains level 2A products, then "level2A" must be set to True. If not, then satellite images must have band number in their name. Example: image1_02.tif.
    resolution : string
        Target resolution for stacked image. Possible options are: 10m, 20m and 60m. Defaults to 10m.
    level2A : bool
        Determines whether the folder path is level 2A product. Defaults to False.
        
    Example:
        stack_images(bands=[2,3,4,5,11,12,'SCL'], folder_path='mosaic')
    '''
    try:
        if str(resolution.replace(' ', '')) not in ['10m', '20m', '60m']:
            raise ValueError
    except ValueError:
        print('ValueError: Resoultion not supported. Possible options are: 10m, 20m or 60m')
        sys.exit(0)

    stacked_dir = _create_folder('stacked')

    if level2A:
        images_to_stack = _find_band_resolution(folder_path, bands, resolution)

    else:
        images_to_stack = []
        for image in os.listdir(folder_path):
            for band in bands:
                # if len(str(band)) > 2:
                if 'B' + str(band) in image or 'B0' + str(band) in image or 'SCL' in image:
                    try:
                        with rasterio.open(os.path.join(folder_path, image)) as src:
                            if os.path.join(folder_path, image) in images_to_stack:
                                continue
                            images_to_stack.append(os.path.join(folder_path, image))
                        continue
                    except:
                        pass

    _stack_and_save(images_to_stack, resolution, stacked_dir)


def _stack_and_save(images, resolution, stack_dir):

    resolution_found = False

    try:
        for image in images:
            with rasterio.open(image) as src0:
                if int(src0.transform[0]) == int(resolution[:-1]):
                    resolution_found = True
                    out_meta = src0.meta.copy()
                    target_width = src0.width
                    target_height = src0.height
                    src_dtype = src0.meta['dtype']
                    break
        if not resolution_found:
            raise FileNotFoundError
    except FileNotFoundError:
        print('No image with {} resolution'.format(resolution))
        sys.exit(0)

    out_meta.update({"driver": "GTiff",
                     "count": len(images),
                     "tiled": True,
                     }
                    )
    stacked_bands = re.findall('[B,A,S,T,W]\w\w', ''.join(images))
    if len(stacked_bands) == 0:
        stack_layer_name = 'stack_{}.tif'.format(len(images))
    else:
        stack_layer_name = 'stack_{}.tif'.format('_'.join(stacked_bands))

    with rasterio.open(os.path.join(stack_dir, stack_layer_name), 'w', **out_meta) as dst:
        print("Stacking bands...")
        for id, layer in enumerate(images, start=1):
            with rasterio.open(layer) as src1:
                if src1.width != target_width or src1.height != target_height:
                    print('Different size')
                    resampled_image = _resample_image(
                        src1, target_width, target_height)
                    dst.write_band(id, resampled_image.astype(src_dtype))
                else:
                    dst.write_band(id, src1.read(1).astype(src_dtype))


def _resample_image(image_src, target_height, target_width):

    print('Resampling..')
    resampled_image = image_src.read(
        out_shape=(
            image_src.count,
            target_height,
            target_width
        ),
        resampling=Resampling.bilinear
    )
    return np.squeeze(resampled_image)