import sys
from DataSets import *
from MalardClient.MalardClient import MalardClient
from MalardClient.DataSet import DataSet
from MalardClient.BoundingBox import BoundingBox
from MalardClient.DataSetQuery import DataSetQuery

import geopandas as gp
from shapely.geometry import Polygon, Point
import os
import numpy as np
import json
from pandas.io.json import json_normalize

# ALASKA
#"outputFileName": "alaska.gpkg",
#"inputDataSet": "ReadyDataAlaska2",
#"runName": "AlaskaRun2",
#"region":"alaska"
#{'column':'within_DataSet','op':'gt','threshold':1}

# HMA
#"outputFileName": "himalayas.json",
#"inputDataSet": "ReadyHim2",
#"runName": "RunHim2",
#"region":"himalayas",
#{'column':'within_DataSet','op':'gt','threshold':1}


class RegressionRun:


    # __conf = {
    #     "outputFileName": "himalayas-gridcells.gpkg",
    #     "inputDataSet": "HimMad2",
    #     "runName": "HimMad2",
    #     "region":"himalayas",
    #     "parentDsName": "mtngla",
    #     "outputPath": "regression_results",
    #     "malardEnvironmentName": "DEVv2",
    #     "malardSyncURL": "http://localhost:9000",
    #     "malardAsyncURL": "ws://localhost:9000",
    #    "filters" : [{'column':'power','op':'gt','threshold':10000},{'column':'coh','op':'gt','threshold':0.6}, \
    #                 {'column':'demDiff','op':'lt','threshold':100}, {'column':'demDiffMadNew','op':'lt','threshold':10}, \
    #                 {'column':'demDiff','op':'gt','threshold':-100}, \
    #                 {'column':'refDifference','op':'gt','threshold':-150}, {'column':'refDifference','op':'lt','threshold':150}, \
    #                 {'column':'within_DataSet','op':'gt','threshold':1}]
    # }

    __conf = {
        "outputFileName": "alaska-gridcells-double.gpkg",
        "inputDataSet": "AlaskaMad",
        "runName": "AlaskaMad",
        "region":"alaska",
        "parentDsName": "mtngla",
        "outputPath": "regression_results",
        "malardEnvironmentName": "DEVv2",
        "malardSyncURL": "http://localhost:9000",
        "malardAsyncURL": "ws://localhost:9000",
        "filters" : [{'column':'power','op':'gt','threshold':10000},{'column':'coh','op':'gt','threshold':0.6}, \
                     {'column':'demDiff','op':'lt','threshold':100}, {'column':'demDiffMadNew','op':'lt','threshold':10}, \
                     {'column':'demDiff','op':'gt','threshold':-100}, \
                     {'column':'refDifference','op':'gt','threshold':-150}, {'column':'refDifference','op':'lt','threshold':150}, \
                     {'column':'within_DataSet','op':'gt','threshold':1}]
    }


    # __conf = {
    #     "outputFileName": "iceland5.gpkg",
    #     "inputDataSet": "tdx",
    #     "runName": "RunIce",
    #     "region":"iceland",
    #     "parentDsName": "mtngla",
    #     "outputPath": "regression_results",
    #     "malardEnvironmentName": "DEVv2",
    #     "malardSyncURL": "http://localhost:9000",
    #     "malardAsyncURL": "ws://localhost:9000",
    #     "filters" : [{'column':'powerScaled','op':'gt','threshold':10000},{'column':'coh','op':'gt','threshold':0.8}, \
    #                   {'column':'demDiff','op':'lt','threshold':200}, {'column':'demDiffMadNew','op':'lt','threshold':40}, \
    #                   ]
    # }




    def __init__(self, logFile=None, notebook=False):
        '''

        :param logFile: if logfile is specified logger will write into file instead of the terminal
        '''
        if logFile is None:
            logging.basicConfig(format='%(asctime)s, %(threadName)s %(thread)d: %(name)s %(levelname)s %(message)s',
                                datefmt='%H:%M:%S', level=logging.INFO)
        else:
            logging.basicConfig(filename=logFile, filemode='a', format='%(asctime)s, %(threadName)s %(thread)d: %(name)s %(levelname)s %(message)s',
                                datefmt='%H:%M:%S', level=logging.INFO)
        sys.excepthook = self.uncaughtErrorHandler

        self.logger = logging.getLogger(__name__)
        self.logger.info('Process started')

        #self.parentDsName = self.config('parentDsName')
        self.inputDataSet = DataSet(parentDs=self.config('parentDsName'), dataSet=self.config('inputDataSet'), region=self.config('region'))
        #self.region = self.config('region')
        self.runName = self.config('runName')

        self.client = MalardClient(notebook=notebook)


        self.query_sync = DataSetQuery(self.config('malardSyncURL'), self.config('malardEnvironmentName'))
        #self.query_async = AsyncDataSetQuery.AsyncDataSetQuery(self.config('malardAsyncURL'), self.config('malardEnvironmentName'), False)
        # get projection
        #self.projection = json.loads(self.client.getProjection(self.parentDsName, self.region))['proj4']
        self.projection = self.client.getProjection(self.inputDataSet).proj4



    def gridcellRegression(self, boundingBox, linear=True, robust=True, weighted=None, minCount=10, radius=None, filters=None):
        if filters is None:
            filters = self.config('filters')
        self.logger.info("Filtering dataset=%s with criteria %s" % (self.inputDataSet, filters))

        result = self.client.executeQuery(self.inputDataSet, boundingBox, projections=[], filters=filters)
        #result = self.client.executeQuery(self.inputDataSet, boundingBox, projections=[])

        self.logger.info("Result message: status=%s, message=%s" % (result.status, result.message))
        data = PointDataSet(result.resultFileName, self.projection)
        self.logger.info("Dataset has %s points" % (data.data.shape[0]))

        if radius is not None:
            centerX=boundingBox.minX+(abs(boundingBox.maxX - boundingBox.minX)/2)
            centerY=boundingBox.minY+(abs(boundingBox.maxY - boundingBox.minY)/2)
            self.logger.info("Apply radius with centerX=%s and centerY=%s..." % (centerX, centerY))
            self.logger.info("Before radius count=%s..." % (data.data.shape[0]))
            data.applyRadius(radius=radius, centerX=centerX, centerY=centerY)
            self.logger.info("After radius count=%s..." % (data.data.shape[0]))

        # release cache of file
        self.client.releaseCacheHandle(result.resultFileName)
        results = {}
        if data.data.shape[0]>minCount and not data.data['time'].nunique()==1:
            if linear:
                r = data.linearRegression()
                results = {**results, **r}
            if robust:
                r=data.robustRegression()
                results = {**results, **r}
            if weighted is not None:
                for w in weighted:
                    r= data.weightedRegression(weight=w['weight'], mask=w['mask_std_dev'])
                    results = {**results, **r}
            self.logger.info(results)
        else:
            self.logger.info("Not enough data in cell (%s points)" % (data.data.shape[0]))

        return results



    def regressionFromStats(self, linear=True, robust=True, weighted=None, minT=None, maxT=None, minCount=50, save=True):
        self.logger.info("Get run statistics for parentDS=%s runName=%s ..." % (self.inputDataSet.parentDataSet, self.runName))
        stats = self.query_sync.getRunStatistics(self.inputDataSet.parentDataSet, self.runName)
        stats = json.loads(stats)
        dfStats = json_normalize(stats)
        if minT is None and maxT is None:
            bbx = self.client.boundingBox(self.inputDataSet)
            minT = bbx.minT
            maxT = bbx.maxT

        for idx, line in dfStats.iterrows():
            if line['statistics.afterGlacierMask'] > minCount:
                minX,maxX=line['gridCell.minX'],line['gridCell.minX']+line['gridCell.size']
                minY,maxY=line['gridCell.minY'],line['gridCell.minY']+line['gridCell.size']

                self.logger.info("Calculating gridcell minX=%s maxX=%s minY=%s maxY=%s minT=%s maxT=%s ..." % (minX, maxX, minY, maxY, minT, maxT))

                bbx_in = BoundingBox(minX, maxX, minY, maxY, minT, maxT)

                results = self.gridcellRegression(bbx_in, linear=linear, robust=robust, weighted=weighted)
                self.logger.info("Adding regression results to stats...")
                for key in results:
                    if isinstance(results[key], list):
                        if not np.isin(key, dfStats.columns):
                            newColumn = [key]
                            #
                            dfStats = dfStats.reindex(columns=np.append( dfStats.columns.values, newColumn))
                            dfStats[[key]]= dfStats[[key]].astype('object', inplace=True)
                            dfStats.at[idx, key] = results[key]
                    else:
                        dfStats.at[idx, key] = results[key]

        size = dfStats['gridCell.size']
        geometry = [Point(xy) for xy in zip(dfStats['gridCell.minX']+(size/2), dfStats['gridCell.minY']+(size/2))]
        dfStats = gp.GeoDataFrame(dfStats, crs=self.projection, geometry=geometry)

        if save:
            file = os.path.join(self.config("outputPath"), self.config("outputFileName"))
            self.logger.info("Saving results under file=%s" % file)
            dfStats.to_file(file, driver="GPKG")

        return dfStats


    def regressionFromList(self, gridcells, linear=True, robust=True, weighted=None, minT=None, maxT=None, save=True, radius=None, geometry='point'):

        dfStats = pd.DataFrame(gridcells)

        if minT is None and maxT is None:
            bbx = self.client.boundingBox(self.inputDataSet)
            minT = bbx.minT
            maxT = bbx.maxT

        for idx, line in dfStats.iterrows():

            self.logger.info("Calculating gridcell minX=%s maxX=%s minY=%s maxY=%s minT=%s maxT=%s ..." % (line['minX'],line['maxX'], line['minY'], line['maxY'], minT, maxT))
            bbx_in = BoundingBox(line['minX'].item(), line['maxX'].item(), line['minY'].item(), line['maxY'].item(), minT, maxT)

            results = self.gridcellRegression(bbx_in, linear=linear, robust=robust, weighted=weighted, radius=radius)

            self.logger.info("Adding regression results to stats...")
            for key in results:
                if isinstance(results[key], list):
                    if not np.isin(key, dfStats.columns):
                        newColumn = [key]
                        #
                        dfStats = dfStats.reindex(columns=np.append( dfStats.columns.values, newColumn))
                        dfStats[[key]]= dfStats[[key]].astype('object', inplace=True)
                        dfStats.at[idx, key] = results[key]
                else:
                    dfStats.at[idx, key] = results[key]

        size = dfStats['maxX']-dfStats['minX']
        if geometry=='point:':
            self.logger.info("Converted to point geometry")
            geometry = [Point(xy) for xy in zip(dfStats['minX']+(size/2), dfStats['minY']+(size/2))]
        elif geometry == 'cell':
            self.logger.info("Converted to cell geometry")
            geometry = []
            for idx, line in dfStats.iterrows():
                minX,maxX=line['minX'],line['maxX']
                minY,maxY=line['minY'],line['maxY']
                geometry.append(Polygon([(minX,minY), (minX,maxY), (maxX,maxY), (maxX,minY), (minX,minY)]))
        else:
            self.logger.info("Error: not valid geometry specified. Should be either 'point' or 'cell'")
        dfStats = gp.GeoDataFrame(dfStats, crs=self.projection, geometry=geometry)

        if save:
            file = os.path.join(self.config("outputPath"), self.config("outputFileName"))
            self.logger.info("Saving results under file=%s" % file)
            dfStats.to_file(file, driver="GPKG")

        return dfStats


    def regressionFromRaster(self, file, linear=True, robust=True, weighted=None, minT=None, maxT=None, save=True, rasterNoData=-1000000, radius=None, geometry='point'):
        ''' Calcualtes regression from cells corresponding to the cells of a given input raster

        :param file: rasterfile path
        :param radius: if None the exact extent of the raster cell is used, else the center of the rastercell and the points within a given rasius is used
        :return:
        '''
        self.logger.info("Start regression from raster for parentDS=%s runName=%s ..." % (self.inputDataSet.parentDataSet, self.runName))
        if minT is None and maxT is None:
            bbx = self.client.boundingBox(self.inputDataSet)
            minT = bbx.minT
            maxT = bbx.maxT

        raster = RasterDataSet(file)

        if radius is None:
            extents = raster.getCellsAsExtent()
        else:
            xy, values = raster.getCenterPoints()
            extents=[]
            for i,el in enumerate(xy):
                self.logger.info("Calculating gridcell %s / %s ..." % (i+1, len(values)))
                if values[i] != rasterNoData:
                    ext = {'minX':el[0]-radius, 'maxX': el[0]+radius, 'minY':el[1]-radius, 'maxY':el[1]+radius}
                    extents.append(ext)
                    self.logger.info("Extent with radius=%s is minX=%s maxX=%s minY=%s maxY=%s ..." % (radius, ext['minX'], ext['maxX'], ext['minY'], ext['maxY']))
                else:
                    self.logger.info("Raster cell=%s has no data no (datavalue=%s) and is skipped  ..." % (el, rasterNoData))

        stats = self.regressionFromList(extents, linear=linear, robust=robust, weighted=weighted, minT=minT, maxT=maxT, save=save, radius=radius, geometry=geometry)

        return stats


    def regressionFromFile(self, file, linear=True, robust=True, weighted=None, minT=None, maxT=None, save=True, radius=None, geometry='point'):
        ''' Calcualtes regression from cells corresponding to the cells of a given input raster

        :param file: rasterfile path
        :param radius: if None the exact extent of the raster cell is used, else the center of the rastercell and the points within a given rasius is used
        :return:
        '''

        self.logger.info("Start regression from file for parentDS=%s ..." % (self.inputDataSet.parentDataSet))
        if minT is None and maxT is None:
            bbx = self.client.boundingBox(self.inputDataSet)
            minT = bbx.minT
            maxT = bbx.maxT

        extents =[]
        with open(file) as f:
            for line in f:
                split = line.strip().split(",")
                ext = {'minX':int(split[0]), 'maxX': int(split[1]), 'minY':int(split[2]), 'maxY':int(split[3])}
                extents.append(ext)
        stats = self.regressionFromList(extents, linear=linear, robust=robust, weighted=weighted, minT=minT, maxT=maxT, save=save, radius=radius, geometry=geometry)

        return stats



    @staticmethod
    def config(name):
        return RegressionRun.__conf[name]
    def uncaughtErrorHandler(self, type, value, tb):
        self.logger.error("Uncaught exception", exc_info=(type, value, tb))



if __name__ ==  '__main__':
    #(500000, 600000, 0, 100000)
    #(700000, 800000, 0, 100000)
    #(500000, 600000, 100000, 200000)
    #(500000, 600000, -100000, 0)

    # reg = RegressionRun()
    # #minT and maxT
    # bbx = reg.client.boundingBox(reg.inputDataSet)
    # minT = bbx.minT
    # maxT = bbx.maxT
    # # minX etc.
    # minX = 400000
    # maxX = 500000
    # minY = 0
    # maxY = 100000
    # # minX = -1200000
    # # maxX = -1100000
    # # minY = 600000
    # # maxY = 700000
    #
    # bbx_in = BoundingBox(minX, maxX, minY, maxY, minT, maxT)
    # #
    # results = reg.gridcellRegression(bbx_in, robust=True, linear=True, weighted=[{'weight':'power', 'mask_std_dev':3},{'weight':'coh', 'mask_std_dev':3},{'weight':'powercoh', 'mask_std_dev':3}])
    # print(results)


    # RUN ALL
    #reg = RegressionRun()
    #reg.regressionFromStats(robust=True, linear=True, weighted=[{'weight':'power', 'mask_std_dev':3},{'weight':'coh', 'mask_std_dev':3},{'weight':'powercoh', 'mask_std_dev':3}])


    #mtngla.startProcess()

    # REGRESSION FROM RASTER
    #raster = '/home/livia/IdeaProjects/malard/python/tile_0_1000_1000_0.tif'
    #raster = 'tile_516400_27400_517400_26600.tif'
    #raster = '/data/puma1/scratch/DEMs/Iceland/proj4/rate-small4.tif'

    #raster = '/home/livia/IdeaProjects/malard/python/tile_-45000_-68000_-42000_-71500.tif'
    #2010-12-02 07:25:15, maxT=2019-04-27
    #reg.regressionFromRaster(raster, robust=True, linear=True, radius=500, weighted=[{'weight':'powerScaled', 'mask_std_dev':3},{'weight':'coh', 'mask_std_dev':3}])

    # RUN FROM FILE
    file = 'alaska-gridcells-double.txt'
    reg = RegressionRun()
    reg.regressionFromFile(file=file, robust=True, linear=True, weighted=[{'weight':'power', 'mask_std_dev':3},{'weight':'coh', 'mask_std_dev':3},{'weight':'powercoh', 'mask_std_dev':3}], geometry='cell')