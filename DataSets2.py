
from MalardClient.MalardClient import MalardClient
from MalardClient.DataSet import DataSet
from MalardClient.BoundingBox import BoundingBox
from MalardClient.MaskFilter import MaskFilter

import logging
import pandas as pd
import geopandas as gp
from shapely.geometry import Point, Polygon
import shapely.speedups
from osgeo import gdal
from scipy.interpolate import griddata
import numpy as np
import affine
import statsmodels.api as sm
from scipy.interpolate import griddata
import pandas as  pd
import json
import math
from pandas.io.json import json_normalize
from shapely.geometry import Polygon, Point
import os
import numpy as np
import datetime
from dateutil.relativedelta import *
from MalardClient import MalardHelpers



class PointDataSet:
    def __init__(self, data, projection, stats={}):
        '''
        :param data: can be a dataframe or a filename to a netcdf
        :param projection:
        '''
        self.logger = logging.getLogger(__name__)
        self.projection = projection
        self.stats = stats
        if isinstance(data, pd.DataFrame) or isinstance(data, gp.GeoDataFrame):
            self.data = data
        else:
            self._readData(data)

    def _readData(self, filename):
        if os.path.isfile(filename):
            self.data = MalardHelpers.getDataFrameFromNetCDF(filename)
        else:
            self.logger.info('File=%s empty. Empty dataframe is created', filename)
            self.data = pd.DataFrame()


    def applyRadius(self, radius, centerX, centerY):
        self.logger.info("Apply radius =%s with centerX=%s and centerY=%s" % (radius, centerX,centerY))
        # pythagoras :)
        distX = abs(self.data.x-centerX)
        distY = abs(self.data.y-centerY)
        self.data['distanceToCenter']= np.sqrt((distX*distX) + (distY*distY))
        dfRad = self.data[self.data['distanceToCenter']<=radius]
        self.data = dfRad
        return self.data

    def asGeoDataSet(self):
        if hasattr(self, 'geoDataSet'):
            return self.geoDataSet
        else:
            geoDf = self._df_to_geodf()
            geoDs = PointGeoDataSet(geoDf, self.projection, stats=self.stats)
            self.geoDataSet = geoDs
            return self.geoDataSet


    def _df_to_geodf(self):
        '''Helper function to convert a pandas dataframe to a geopandas geodataframe'''
        if not self.data.empty:
            shapely.speedups.enable()
            self.logger.info("Convert DataFrame to GeoDataFrame... ")
            # Convert points to geometries
            geometry = [Point(xy) for xy in zip(self.data.x, self.data.y)]
            geoDf = gp.GeoDataFrame(self.data, crs=self.projection, geometry=geometry)
            self.logger.info("Conversion to GeoDataFrame successful!")
            return geoDf
        else:
            return gp.GeoDataFrame()

    def calculateElevationDifference(self, dem, buffer=0):
        '''
        :param dem: a RasterDataSet
        :param buffer: buffer in meters
        :return:
        '''
        if self.hasData():
            dem.cutToBbx(self.data['x'].min(), self.data['x'].max(), self.data['y'].min(), self.data['y'].max(), buffer)
            xy, values = dem.getCenterPoints()
            # get point coordinates as numpy
            #coords = np.asarray([self.data['x'], self.data['y']])
            coords = np.vstack((self.data['x'].values, self.data['y'].values))
            coords = np.transpose(coords)

            # interpolate
            self.logger.info("Interpolating... ")
            self.data['refElevation'] = griddata(xy, values, coords, method='cubic')

            # calculate difference
            self.data['refDifference'] = self.data['elev']-self.data['refElevation']
            self.logger.info("Finished calculating difference, mean difference =  %s... " % self.data['refDifference'].mean())
            self.stats['meanElevationDifference'] = self.data['refDifference'].mean()


    def linearRegression(self):
        self.logger.info("Linear regression...")
        # model variables
        vals = np.asarray([self.data.time, self.data.x, self.data.y])
        vals = np.transpose(vals)
        vals = sm.add_constant(vals)
        elev = self.data.elev

        # Create model and fit it (least squares)
        model = sm.OLS(elev, vals)
        # OR robust model -- note that it won't have r squared
        #model = sm.RLM(y, x)
        results = model.fit()

        regression_results = {}

        regression_results['regression.rsquared'] = results.rsquared
        regression_results['regression.c'] = results.params.x1
        regression_results['regression.c.se'] = results.bse.x1
        regression_results['regression.c.year'] = results.params.x1*31536000
        regression_results['regression.c.se.year'] = results.bse.x1*31536000
        regression_results['regression.const'] = results.params.const
        regression_results['regression.const.se'] = results.bse.const
        regression_results['regression.count'] = results.nobs

        regression_results['regression.const.pvalue'] = results.pvalues.const
        regression_results['regression.c.pvalue'] = results.pvalues.x1
        regression_results['regression.const.tvalue'] = results.tvalues.const
        regression_results['regression.c.tvalue'] = results.tvalues.x1

        intervals = results.conf_int(alpha=0.05, cols=None)
        regression_results['regression.c.conf_interval.low'] = intervals[intervals.index=='x1'].values[0][0]*31536000
        regression_results['regression.c.conf_interval.high'] = intervals[intervals.index=='x1'][1].values[0]*31536000
        regression_results['regression.const.conf_interval.low'] = intervals[intervals.index=='const'][0].values[0]
        regression_results['regression.const.conf_interval.high'] = intervals[intervals.index=='const'][1].values[0]

        if hasattr(self, 'regression_results'):
            merge = {**self.regression_results, **regression_results}
            self.regression_results = merge
        else:
            self.regression_results = regression_results

        return regression_results


    def weightedRegression(self, weight='ones', mask=None):
        '''

        :param weight:
        :param mask: If true all values with
        :return:
        '''
        self.logger.info("Weighted regression...")
        # model variables
        vals = np.asarray([self.data.time, self.data.x, self.data.y])
        vals = np.transpose(vals)
        vals = sm.add_constant(vals)
        elev = self.data.elev

        if weight =='ones':
            self.logger.info("Weighted regression with ones as weights...")
            weights = np.ones(self.data.shape[0])
        else:
            self.logger.info("Weighted regression with {} as weights...".format(weight))
            # weights according to script
            w = self.data[weight]*self.data[weight]
            w = w/max(w)
            weights = w*w

        if isinstance(mask, int):
            self.logger.info("Mask out points with > {} x Standard deviation...".format(mask))
            mask = abs(elev-np.median(elev))>(mask*np.std(elev))
            weights = np.where(mask,0,w)


        # Create model and fit it (least squares)
        model = sm.WLS(elev, vals, weights)

        results = model.fit()

        regression_results = {}

        regression_results['regression.w_{}.rsquared'.format(weight)] = results.rsquared
        regression_results['regression.w_{}.c'.format(weight)] = results.params.x1
        regression_results['regression.w_{}.c.se'.format(weight)] = results.bse.x1
        regression_results['regression.w_{}.c.year'.format(weight)] = results.params.x1*31536000
        regression_results['regression.w_{}.c.se.year'.format(weight)] = results.bse.x1*31536000
        regression_results['regression.w_{}.const'.format(weight)] = results.params.const
        regression_results['regression.w_{}.const.se'.format(weight)] = results.bse.const
        regression_results['regression.w_{}.count'.format(weight)] = np.count_nonzero(abs(weights))
        regression_results['regression.w_{}.count_masked'.format(weight)] = results.nobs-np.count_nonzero(abs(weights))

        regression_results['regression.w_{}.const.pvalue'] = results.pvalues.const
        regression_results['regression.w_{}.c.pvalue'] = results.pvalues.x1
        regression_results['regression.w_{}.const.tvalue'] = results.tvalues.const
        regression_results['regression.c.tvalue'] = results.tvalues.x1

        intervals = results.conf_int(alpha=0.05, cols=None)
        regression_results['regression.w_{}.c.conf_interval.low'] = intervals[intervals.index=='x1'].values[0][0]*31536000
        regression_results['regression.w_{}.c.conf_interval.high'] = intervals[intervals.index=='x1'][1].values[0]*31536000
        regression_results['regression.w_{}.const.conf_interval.low'] = intervals[intervals.index=='const'][0].values[0]
        regression_results['regression.w_{}.const.conf_interval.high'] = intervals[intervals.index=='const'][1].values[0]

        if hasattr(self, 'regression_results'):
            merge = {**self.regression_results, **regression_results}
            self.regression_results = merge
        else:
            self.regression_results = regression_results

        return regression_results


    def timeSeries(self, startdate=datetime.datetime(2010,11,1,0,0), enddate=datetime.datetime(2019,1,1,0,0), interval=3):
        self.logger.info("Calculating time series...")

        dateobjects = []
        for i, row in self.data.iterrows():
            date = datetime.datetime.utcfromtimestamp(self.data.time[i])
            dateobjects.append(date)

        self.data['dateobject'] = dateobjects

        averages = []
        dates = []
        changes = []
        medians = []

        use_date = startdate
        while use_date <= enddate:
            df_filt = self.data[(self.data.dateobject >= use_date) & (self.data.dateobject <(use_date+relativedelta(months=+interval)))]
            averages.append(df_filt.refDifference.mean())
            medians.append(df_filt.refDifference.median())
            dates.append(use_date)
            use_date = use_date+relativedelta(months=+interval)

        for i in averages:
            changes.append(i-averages[0])

        timeseries_results = {}

        timeseries_results['timeseries.dates'] = dates
        timeseries_results['timeseries.averages'] = averages
        timeseries_results['timeseries.medians'] = medians
        timeseries_results['timeseries.change'] = changes

        if hasattr(self, 'timeseries_results'):
            merge = {**self.timeseries_results, **timeseries_results}
            self.timeseries_results = merge
        else:
            self.timeseries_results = timeseries_results
        return timeseries_results


    def robustRegression(self):
        self.logger.info("Robust regression...")
        # model variables


        vals = np.asarray([self.data.time, self.data.x, self.data.y])
        vals = np.transpose(vals)
        vals = sm.add_constant(vals)
        elev = self.data.elev

        # Create model and fit it (least squares)
        model = sm.RLM(elev, vals)
        results = model.fit()

        regression_results = {}

        regression_results['regression.robust.c'] = results.params.x1
        regression_results['regression.robust.c.se'] = results.bse.x1
        regression_results['regression.robust.c.year'] = results.params.x1*31536000
        regression_results['regression.robust.c.se.year'] = results.bse.x1*31536000
        regression_results['regression.robust.const'] = results.params.const
        regression_results['regression.robust.const.se'] = results.bse.const
        regression_results['regression.robust.count'] = results.nobs

        regression_results['regression.robust.const.pvalue'] = results.pvalues.const
        regression_results['regression.robust.c.pvalue'] = results.pvalues.x1
        regression_results['regression.robust.const.tvalue'] = results.tvalues.const
        regression_results['regression.robust.c.tvalue'] = results.tvalues.x1

        intervals = results.conf_int(alpha=0.05, cols=None)
        regression_results['regression.robust.c.conf_interval.low'] = intervals[intervals.index=='x1'].values[0][0]*31536000
        regression_results['regression.robust.c.conf_interval.high'] = intervals[intervals.index=='x1'][1].values[0]*31536000
        regression_results['regression.robust.const.conf_interval.low'] = intervals[intervals.index=='const'][0].values[0]
        regression_results['regression.robust.const.conf_interval.high'] = intervals[intervals.index=='const'][1].values[0]

        if hasattr(self, 'regression_results'):
            merge = {**self.regression_results, **regression_results}
            self.regression_results = merge
        else:
            self.regression_results = regression_results

        return regression_results


    def hasData(self):
        '''Returns True if dataframe not empty, returns false if empty'''
        return not self.data.empty

    def addStatistic(self, key, value):
        '''adds a key - value pair to statistics

        :param key:
        :param value:
        :return:
        '''
        self.stats[key] = value

    def getStats(self):
        return self.stats
    def length(self):
        return float(len(self.data.index))
    def mean(self, column):
        return float(self.data[column].mean())
    def min(self, column):
        return self.data[column].min()
    def max(self, column):
        return self.data[column].max()

class PointGeoDataSet(PointDataSet):

    def applyMask(self, maskPath, maskType):
        '''Filters out data points that aren't inside the mask

        :param maskPath: path to mask file
        :param maskType: mask type, e.g. "Glacier" or "Debris"
        :return:
        '''
        if self.hasData():
            self.logger.info("Read %s mask file... " % maskType)
            if os.path.exists(maskPath):
                mask = gp.read_file(maskPath)

                self.stats['%sMaskArea'%maskType] = float(mask['area'].sum())
                # drop all columns except geometry
                mask.drop(mask.columns.difference(['geometry']), 1, inplace=True)

                self.logger.info("Apply %s filter mask... (deletes points which aren't within masks) " % maskType)
                maskedGla=gp.sjoin(self.data, mask, how='inner', op='within')
                maskedGla.drop(columns=['index_right'], axis=1, inplace=True)

                self.stats['after%sMask' % maskType] = float(len(maskedGla.index))
                self.logger.info("After applying %s mask: point count [%d]" % (maskType,len(maskedGla.index)))
                self.data = maskedGla
            else:
                self.logger.error('Error: File path=%s of type=%s is invalid' % (maskPath, maskType))


    def withinMask(self, maskPath, maskType):
        '''Adds a column to data decribing if point is within mask or not
        0 means not within mask
        1 means within mask

        :param maskPath:
        :param maskType:
        :return:
        '''

        if self.hasData():
            self.logger.info("Read %s mask file... " % maskType)
            if os.path.exists(maskPath):
                if maskPath.endswith('.tif') or maskPath.endswith('.tiff'):
                    self._withinMaskRaster(maskPath, maskType)
                else:
                    self._withinMaskPolys(maskPath, maskType)
            else:
                self.logger.error('Error: File path=%s of type=%s is invalid' % (maskPath, maskType))
                self.data['within_%s' % maskType] = 0
                self.stats['pointsWithin%sMask' % maskType] = 0.0


    def _withinMaskPolys(self, maskPath, maskType):
        self.logger.info("Read polygon mask file... ")
        mask = gp.read_file(maskPath)

        self.stats['%sMaskArea'%maskType] = float(mask['area'].sum())
        # drop all columns except geometry
        mask.drop(mask.columns.difference(['geometry']), 1, inplace=True)

        self.logger.info("Apply %s mask (adds a column to points describing if they within mask)... " % maskType)
        masked = gp.sjoin(self.data, mask, how='left', op='within')

        # assign 1 to points that are inside debris mask and 0 to points that are not inside
        masked['within_%s' % maskType] = (masked['index_right'] >= 0).astype(int)
        masked.drop('index_right', axis=1, inplace=True)

        # summary
        count = masked.loc[(masked['within_%s' % maskType] == 1)].shape[0]
        self.stats['pointsWithin%sMask' % maskType] = float(count)
        self.logger.info("Points within %s mask: count [%d]" % (maskType, count))
        self.data = masked
        return ''

    def _withinMaskRaster(self, maskPath, maskType):
        self.logger.info("Read raster mask file... ")
        raster = RasterDataSet(maskPath)
        buffer = raster.data.GetGeoTransform()[1]*2
        raster.cutToBbx(self.data['x'].min(), self.data['x'].max(), self.data['y'].min(), self.data['y'].max(), buffer=buffer)

        self.logger.info("Apply %s mask (adds a column to points describing if they within mask)... " % maskType)
        values = raster.getValuesAt(self.data['x'].tolist(), self.data['y'].tolist())
        self.data['within_%s' % maskType] = values

        # summary
        unique = self.data['within_%s' % maskType].unique()

        for i in unique:
            count = self.data.loc[(self.data['within_%s' % maskType] == i)].shape[0]
            ratio = (float(count)/ float(self.data.shape[0])) *100.0
            self.stats['pointsOn%sValue%s' % (maskType, int(i))] = float(count)
            self.stats['pointsOn%sValue%sRatio' % (maskType, int(i))] = ratio
            self.logger.info("Points within %s mask value %s: count [%d]" % (maskType, i, count))
            self.logger.info("Points within %s mask value %s: ratio [%s]" % (maskType, i, ratio))
        #count = self.data.loc[(self.data['within_%s' % maskType] == 1)].shape[0]
        #ratio = (float(count)/ float(self.data.shape[0])) *100.0
        #self.stats['pointsWithin%sMask' % maskType] = float(count)
        #self.stats['pointsWithin%sMaskRatio' % maskType] = ratio
        #self.stats['pointsNotWithin%sMaskRatio' % maskType] = 100.0-ratio
        #self.logger.info("Points within %s mask: count [%d]" % (maskType, count))
        #self.logger.info("Points within %s mask: ratio [%s]" % (maskType, ratio))


    def getData(self, geo=True):
        '''Returns data
        if  geo is False it removes the geo references and shapes'''
        self.logger.info('Converting geodataframe to dataframe...')
        if geo is False:
            data = self.data.drop(columns=['geometry'], axis=1) #drop geometry column
            return pd.DataFrame(data)
        else:
            return self.data



class RasterDataSet:
    def __init__(self, filename):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Opening raster file:  %s... " % (filename))
        self.data = gdal.Open(filename, gdal.GA_ReadOnly)


    def cutToBbx(self, minX, maxX, minY, maxY, buffer=0):
        ''' Cuts dataset to boundingbox in memory, note that data outside bounding box will be lost

        :param minX:
        :param maxX:
        :param minY:
        :param maxY:
        :param buffer:
        :return:
        '''
        out_tif = "/vsimem/tile_%s_%s_%s_%s.tif" % (minX-buffer, maxY+buffer, maxX+buffer, minY-buffer)
        self.logger.info("Saved as tile_%s_%s_%s_%s.tif" % (minX-buffer, maxY+buffer, maxX+buffer, minY-buffer))
        self.logger.info("Clipping raster file to minX=%s maxX=%s minY=%s maxY=%s with buffer=%s... " % (minX, maxX, minY, maxY, buffer))
        self.data = gdal.Translate(out_tif, self.data, projWin = [minX-buffer, maxY+buffer, maxX+buffer, minY-buffer])

    def getValuesAt(self, x, y):
        '''

        :param x: either an array/list or one value
        :param y: either an array/list or one value
        :return: either an array/list or one value
        '''
        #data = self.data.ReadAsArray().astype(np.float)
        data = self.data.GetRasterBand(1).ReadAsArray()
        if isinstance(x, (list, np.ndarray)) and isinstance(y, (list, np.ndarray)):
            values =[]
            for idx, i in enumerate(x):
                values.append(data[self._getPixel(x[idx],y[idx],self.data.GetGeoTransform())])
            return values
        else:
            return data[self._getPixel(x,y,self.data.GetGeoTransform())]


    def _getPixel(self, x, y, gt):
        '''

        :param x:
        :param y:
        :param gt: GeoTransform
        :return:
        '''
        py = int((x-gt[0])/gt[1])
        px = int((y-gt[3])/gt[5])
        return px, py



    def getCenterPoints(self, outEnvironment='/data/puma1/scratch/DEMs/', deleteTemporaryFiles=True):
            '''Returns list with x,y, values of the center of the raster gridcells and their values

            :param outEnvironment: temporary files will be saved here.
            :param deleteTemporaryFiles: if False, temporary files won't be removed
            :return: (coordinates, values) of type (<list>, <list>) with coordinates as 2 dimensional list
            '''
            self.logger.info("Starting to calculate center points of raster... ")
            self.logger.info("Translating raster to XYZ... ")
            out_xyz = os.path.join(outEnvironment, 'tile_%s_%s.xyz' % (self.data.RasterXSize, self.data.RasterYSize)) # file will be deleted after
            gdal.Translate(out_xyz, self.data, format='XYZ', creationOptions=["ADD_HEADER_LINE=YES"])

            self.logger.info("Reading in XYZ file... ")
            xy,values = [], []
            with open(out_xyz, 'r') as f:
                f.readline() # skip first row
                for l in f:
                    row = l.split()
                    xy.append([float(row[0]), float(row[1])])
                    values.append(float(row[2]))
            # remove xyz file
            if deleteTemporaryFiles:
                os.remove(out_xyz)
            return xy, values

    def getCellsAsExtent(self):
        self.logger.info("Starting to calculate extents of all raster cells... ")

        cols = self.data.RasterXSize
        rows = self.data.RasterYSize

        data = self.data.GetRasterBand(1).ReadAsArray()

        extents = []

        #print(self.getCellExtent(self.data.GetGeoTransform(),cols, rows))
        for row in range(0,rows):
            for col in range(0,cols):
                #print(data[row, col], type(data[row, col]))
                extents.append(self.getCellExtent(self.data.GetGeoTransform(),col, row))
        self.logger.info("Finished calculating extents of all raster cells... ")
        return extents

    def getCellExtent(self,gt,cols,rows):
        ''' Return list of corner coordinates from a geotransform

            @type gt:   C{tuple/list}
            @param gt: geotransform
            @type cols:   C{int}
            @param cols: number of columns in the dataset
            @type rows:   C{int}
            @param rows: number of rows in the dataset
            @rtype:    C{[float,...,float]}
            @return:   coordinates with minX, maxX, minY and maxXY        '''
        px = gt[0]+(gt[1]*cols)+(cols*gt[2])
        py = gt[3]+(gt[5]*rows)+(cols*gt[4])

        results = {}
        results['minX'] = px
        results['maxY'] = py
        results['maxX'] = px+gt[1]+gt[2]
        results['minY'] = py+gt[5]+gt[4]
        return results



if __name__ ==  '__main__':
    logging.basicConfig(level=logging.INFO)
    fp = '/data/puma1/scratch/v2/malard/export/mtngla_tdx_1556569735.nc'
    dataSet = 'tdx'
    projection = "+proj=aea +lat_1=25 +lat_2=47 +lat_0=36 +lon_0=85 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"

    glacierMask = '/data/puma1/scratch/malard/mask/mtngla/static/RGIv60/Glacier/HMA/cell_x400000_y0_s100000/mask_Glacier_x400000_y0_s100000.gpkg'
    debrisMask = '/data/puma1/scratch/malard/mask/mtngla/static/SDCv10/Debris/HMA/cell_x400000_y0_s100000/mask_Debris_x400000_y0_s100000.gpkg'
    minX = 400000
    maxX = 500000
    minY = 0
    maxY = 100000
    size = 100000

    referenceDem = "/data/puma1/scratch/DEMs/srtm_test.tif"
    demDataSetMask = "/data/puma1/scratch/mtngla/DEMs-coreg/Tdx_Srtm_SurfaceSplit.tiff"


    raster = RasterDataSet(referenceDem)
    raster.cutToBbx(516400,517400,26600,27400)
    #values = raster.getValuesAt([100,200], [1000,700])
    #values = raster.getValueAt(100, 700)
    #print(values)

    #ds = PointDataSet(fp, projection)

    #geoDs = ds.asGeoDataSet()
    #geoDs.applyMask(glacierMask, 'Glacier')

    #geoDs.calculateElevationDifference(raster, buffer=10000)


    #geoDs.withinMask(debrisMask, 'Debris')

    #geoDs.withinMask(referenceDem, 'DemDataSets')

    #print(geoDs.data['within_DemDataSets'])
    #print(geoDs.data.head())

    #exit(0)
