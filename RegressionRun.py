import sys
from DataSets2 import *
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

# HMA
#"outputFileName": "himalayas.json",
#"inputDataSet": "ReadyHim2",
#"runName": "RunHim2",
#"region":"himalayas",


class RegressionRun:


    # __conf = {
    #     "outputFileName": "himalayas-from-raster.gpkg",
    #     "inputDataSet": "ReadyHim2",
    #     #"inputDataSet": "tdx2",
    #     "runName": "RunHim2",
    #     "region":"himalayas",
    #     "parentDsName": "mtngla",
    #     "outputPath": "regression_results",
    #     "malardEnvironmentName": "DEVv2",
    #     "malardSyncURL": "http://localhost:9000",
    #     "malardAsyncURL": "ws://localhost:9000",
    #    "filters" : [{'column':'power','op':'gt','threshold':10000},{'column':'coh','op':'gt','threshold':0.6}, \
    #                 {'column':'demDiff','op':'lt','threshold':100}, {'column':'demDiffMad','op':'lt','threshold':10}, \
    #                 {'column':'demDiff','op':'gt','threshold':-100}, {'column':'demDiffMad','op':'gt','threshold':-10}, \
    #                 {'column':'refDifference','op':'gt','threshold':-150}, {'column':'refDifference','op':'lt','threshold':150}]
    # }

    __conf = {
        "outputFileName": "iceland.gpkg",
        "inputDataSet": "tdx",
        "runName": "RunIce",
        "region":"iceland",
        "parentDsName": "mtngla",
        "outputPath": "regression_results",
        "malardEnvironmentName": "DEVv2",
        "malardSyncURL": "http://localhost:9000",
        "malardAsyncURL": "ws://localhost:9000",
        "filters" : [{'column':'powerScaled','op':'gt','threshold':10000},{'column':'coh','op':'gt','threshold':0.8}, \
                      {'column':'demDiff','op':'lt','threshold':200}, {'column':'demDiffMad','op':'lt','threshold':40}, \
                      {'column':'demDiff','op':'gt','threshold':-200}, {'column':'demDiffMad','op':'gt','threshold':-40}]
    }




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



    def gridcellRegression(self, boundingBox, linear=True, robust=True, weighted=None, minCount=20):
        filters = self.config('filters')
        self.logger.info("Filtering dataset=%s with criteria %s" % (self.inputDataSet, filters))

        result = self.client.executeQuery(self.inputDataSet, boundingBox, projections=[], filters=filters)
        #result = self.client.executeQuery(self.inputDataSet, boundingBox, projections=[])

        self.logger.info("Result message: status=%s, message=%s" % (result.status, result.message))
        data = PointDataSet(result.resultFileName, self.projection)
        print(data.data.shape)
        print('coh ', data.data.coh.min(), data.data.coh.max())
        print('power ', data.data.powerScaled.min(), data.data.powerScaled.max())
        print('demdiff ', data.data.demDiff.min(), data.data.demDiff.max())
        print('demdiffMad ', data.data.demDiffMad.min(), data.data.demDiffMad.max())
        # release cache of file
        self.client.releaseCacheHandle(result.resultFileName)
        results = {}
        if data.data.shape[0]>minCount:
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


    def regressionFromList(self, gridcells, linear=True, robust=True, weighted=None, minT=None, maxT=None, save=True, radius=None):

        dfStats = pd.DataFrame(gridcells)

        if minT is None and maxT is None:
            bbx = self.client.boundingBox(self.inputDataSet)
            minT = bbx.minT
            maxT = bbx.maxT

        for idx, line in dfStats.iterrows():

            self.logger.info("Calculating gridcell minX=%s maxX=%s minY=%s maxY=%s minT=%s maxT=%s ..." % (line['minX'],line['maxX'], line['minY'], line['maxY'], minT, maxT))

            bbx_in = BoundingBox(line['minX'], line['maxX'], line['minY'], line['maxY'], minT, maxT)

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

        size = dfStats['maxX']- dfStats['minX']
        geometry = [Point(xy) for xy in zip(dfStats['minX']+(size/2), dfStats['minY']+(size/2))]
        dfStats = gp.GeoDataFrame(dfStats, crs=self.projection, geometry=geometry)

        if save:
            file = os.path.join(self.config("outputPath"), self.config("outputFileName"))
            self.logger.info("Saving results under file=%s" % file)
            dfStats.to_file(file, driver="GPKG")

        return dfStats


    def regressionFromRaster(self, file, linear=True, robust=True, weighted=None, minT=None, maxT=None, save=True, rasterNoData=-1000000, radius=None):
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
                if values[i] != rasterNoData:
                    extents.append({'minX':el[0]-radius, 'maxX': el[0]+radius, 'minY':el[1]-radius, 'maxY':el[1]+radius})

        stats = self.regressionFromList(extents, linear=linear, robust=robust, weighted=weighted, minT=minT, maxT=maxT, save=save, radius=radius)

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

    reg = RegressionRun()
    # minT and maxT
    #bbx = reg.client.boundingBox(reg.inputDataSet)
    #minT = bbx.minT
    #maxT = bbx.maxT
    # minX etc.
    #minX = 400000
    #maxX = 500000
    #minY = 0
    #maxY = 100000

    #bbx_in = BoundingBox(minX, maxX, minY, maxY, minT, maxT)

    #results = reg.gridcellRegression(bbx_in)
    #print(results)

    # RUN ALL
    #reg = RegressionRun()
    #reg.regressionFromStats()


    #mtngla.startProcess()

    # REGRESSION FROM RASTER
    #raster = '/home/livia/IdeaProjects/malard/python/tile_0_1000_1000_0.tif'
    #raster = 'tile_516400_27400_517400_26600.tif'
    raster = '/data/puma1/scratch/DEMs/Iceland/rate-small.tif'

    #raster = '/home/livia/IdeaProjects/malard/python/tile_-45000_-68000_-42000_-71500.tif'
    #2010-12-02 07:25:15, maxT=2019-04-27
    reg.regressionFromRaster(raster, robust=True, linear=True,radius=500, weighted=[{'weight':'powerScaled', 'mask_std_dev':3},{'weight':'coh', 'mask_std_dev':3}])

