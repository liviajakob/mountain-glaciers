import sys

from DataSets import *
from MalardClient.MalardClient import MalardClient
from MalardClient.DataSet import DataSet
from MalardClient.BoundingBox import BoundingBox
from MalardClient.DataSetQuery import DataSetQuery
import datetime
import os
import numpy as np
import json
from pandas.io.json import json_normalize

# ALASKA
#"outputFileName": "alaska.json",
#"inputDataSet": "ReadyDataAlaska2",
#"runName": "AlaskaRun2",
#"region":"alaska",

# HMA
#"outputFileName": "himalayas.json",
#"inputDataSet": "ReadyHim2",
#"runName": "RunHim2",
#"region":"himalayas",

class TimeseriesRun:

    __conf = {
        "outputFileName": "himalayas-weighted-tdx.json",
        "inputDataSet": "ReadyHim2",
        "runName": "RunHim2",
        "region":"himalayas",
        "parentDsName": "mtngla",
        "outputPath": "timeseries_results",
        "malardEnvironmentName": "DEVv2",
        "malardSyncURL": "http://localhost:9000",
        "malardAsyncURL": "ws://localhost:9000",
        "filters" : [{'column':'power','op':'gt','threshold':10000},{'column':'coh','op':'gt','threshold':0.6}, \
                     {'column':'demDiff','op':'lt','threshold':100}, {'column':'demDiffMad','op':'lt','threshold':10}, \
                     {'column':'demDiff','op':'gt','threshold':-100}, {'column':'demDiffMad','op':'gt','threshold':-10}, \
                     {'column':'refDifference','op':'gt','threshold':-150}, {'column':'refDifference','op':'lt','threshold':150}, \
                     {'column':'within_DataSet','op':'gt','threshold':1}]
    }

    # __conf = {
    #     "outputFileName": "alaska-weighted-tdx.json",
    #     "runName": "AlaskaRun2",
    #     "inputDataSet": "ReadyDataAlaska2",
    #     "region":"alaska",
    #     "parentDsName": "mtngla",
    #     "outputPath": "timeseries_results",
    #     "malardEnvironmentName": "DEVv2",
    #     "malardSyncURL": "http://localhost:9000",
    #     "malardAsyncURL": "ws://localhost:9000",
    #     "filters" : [{'column':'power','op':'gt','threshold':10000},{'column':'coh','op':'gt','threshold':0.6}, \
    #                  {'column':'demDiff','op':'lt','threshold':100}, {'column':'demDiffMad','op':'lt','threshold':10}, \
    #                  {'column':'demDiff','op':'gt','threshold':-100}, {'column':'demDiffMad','op':'gt','threshold':-10}, \
    #                  {'column':'refDifference','op':'gt','threshold':-150}, {'column':'refDifference','op':'lt','threshold':150}, \
    #                  {'column':'within_DataSet','op':'gt','threshold':1}]
    # }


    def __init__(self, logFile=None):
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

        self.client = MalardClient(notebook=False)


        self.query_sync = DataSetQuery(self.config('malardSyncURL'), self.config('malardEnvironmentName'))
        #self.query_async = AsyncDataSetQuery.AsyncDataSetQuery(self.config('malardAsyncURL'), self.config('malardEnvironmentName'), False)
        # get projection
        #self.projection = json.loads(self.client.getProjection(self.parentDsName, self.region))['proj4']
        self.projection = self.client.getProjection(self.inputDataSet).proj4



    def gridcellTimeseries(self, boundingBox, startdate, enddate, interval, weighted=[]):
        filters = self.config('filters')
        self.logger.info("Filtering dataset=%s with criteria %s" % (self.inputDataSet, filters))
        result = self.client.executeQuery(self.inputDataSet, boundingBox, projections=[], filters=filters)
        self.logger.info("Result message: result=%s, message=%s" % (result.status, result.message))

        data = PointDataSet(result.resultFileName, self.projection)
        # release cache of file
        self.client.releaseCacheHandle(result.resultFileName)
        results = {}
        if data.hasData():
            self.logger.info('Data length={}'.format(data.length()))
            r = data.timeSeries(startdate=startdate, enddate=enddate, interval=interval, weighted=weighted)
            results = {**results, **r}
            self.logger.info(results)
        else:
            self.logger.info('No data in file')


        return results



    def timeseriesFromStats(self, startdate, enddate, interval=3, minT=None, maxT=None, minCount=0, save=True, weighted=None):
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
                self.logger.info("Calculating gridcell minX=%s minY=%s ..." % (minX, maxX))

                bbx_in = BoundingBox(minX, maxX, minY, maxY, minT, maxT)

                results = self.gridcellTimeseries(bbx_in, startdate, enddate, interval, weighted=weighted)
                self.logger.info("Adding timeseries results to stats...")
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
                    else:
                        dfStats.at[idx, key] = results[key]


        #size = dfStats['gridCell.size']
        #geometry = [Point(xy) for xy in zip(dfStats['gridCell.minX']+(size/2), dfStats['gridCell.minY']+(size/2))]
        #dfStats = gp.GeoDataFrame(dfStats, crs=self.projection, geometry=geometry)

        if save:
            file = os.path.join(self.config("outputPath"), self.config("outputFileName"))
            self.logger.info("Saving results under file=%s" % file)
            dfStats.to_json(file)

        return dfStats



    @staticmethod
    def config(name):
        return TimeseriesRun.__conf[name]
    def uncaughtErrorHandler(self, type, value, tb):
        self.logger.error("Uncaught exception", exc_info=(type, value, tb))



if __name__ ==  '__main__':
    #(500000, 600000, 0, 100000)
    #(700000, 800000, 0, 100000)
    #(500000, 600000, 100000, 200000)
    #(500000, 600000, -100000, 0)

    reg = TimeseriesRun()
    # minT and maxT
    bbx = reg.client.boundingBox(reg.inputDataSet)
    minT = bbx.minT
    maxT = bbx.maxT
    # minX etc.
    #minX = -3900000
    #maxX = -3800000
    #minY = -600000
    #maxY = -500000
    #
    minX = 400000
    maxX = 500000
    minY = 0
    maxY = 100000

    #bbx_in = BoundingBox(minX, maxX, minY, maxY, minT, maxT)
    #results = reg.gridcellTimeseries(bbx_in, startdate=datetime.datetime(2010,11,1,0,0), enddate=datetime.datetime(2019,1,1,0,0), interval=3, weighted=[{'weight':'power', 'mask_std_dev':3},{'weight':'coh', 'mask_std_dev':3}])


    #sprint(results)

    # RUN ALL
    reg.timeseriesFromStats(startdate=datetime.datetime(2010,11,1,0,0), enddate=datetime.datetime(2019,1,1,0,0), interval=3, weighted=[{'weight':'power', 'mask_std_dev':3},{'weight':'coh', 'mask_std_dev':3}])
