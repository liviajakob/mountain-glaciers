import logging
import sys
from MalardClient.MalardClient import MalardClient
from MalardClient.DataSet import DataSet
from MalardClient.AsyncDataSetQuery import AsyncDataSetQuery
from MalardClient.DataSetQuery import DataSetQuery
from DataSets import *
import calendar



class MtnGlaGridcellProcess:

    #"referenceDem":"/data/puma1/scratch/DEMs/srtm_test.tif"
    #"referenceDem":"/data/puma1/scratch/mtngla/dems/HMA_TDX_Masked_SRTM_Merged_coreg_aea_clip.tif"

    # HIMALAYAS
    #"runName": "ReadyHim2",
    #"outputDataSet": "Ready8",
    #"parentDsName": "mtngla",
    #"region":"himalayas",
    #"maskDataSet": "RGIv60",
    #"withinDataSets": ["SDCv10", "/data/puma1/scratch/mtngla/dems/Tdx_SRTM_SurfaceSplit.tiff"],
    #"withinDataSetTypes": ["Debris", "DataSet"],
    #"referenceDem": "/data/puma1/scratch/mtngla/dems/HMA_TDX_Masked_SRTM_Merged_coreg_aea_clip.tif",
    #"inputDataSet": "tdx2",
    #
    # "runName": "HimMad2",
    # "outputDataSet": "HimMad2",
    # "parentDsName": "mtngla",
    # "region":"himalayas",


    # ALASKA
    #"runName": "AlaskaRun1",
    #"outputDataSet": "ReadyDataAlaska2",
    #"parentDsName": "mtngla",
    #"region":"alaska",
    #"maskDataSet": "RGIv60",
    #"withinDataSets": ["SDCv10", "/data/puma1/scratch/mtngla/dems/TD_AD_Interp_SurfaceSplit.tiff"],
    #"withinDataSetTypes": ["Debris", "DataSet"],
    #"referenceDem": "/data/puma1/scratch/mtngla/dems/PCR_TdxFilledWithAD_Masked_Polar_Interp_clip.tif",
    #"inputDataSet": "ADwithTDX",


    __conf = {
        "runName": "AlaskaMad",
        "outputDataSet": "AlaskaMad",
        "parentDsName": "mtngla",
        "region":"alaska",
        "maskDataSet": "RGIv60",
        "withinDataSets": ["SDCv10", "/data/puma1/scratch/mtngla/dems/TD_AD_Interp_SurfaceSplit.tiff"],
        "withinDataSetTypes": ["Debris", "DataSet"],
        "referenceDem": "/data/puma1/scratch/mtngla/dems/PCR_TdxFilledWithAD_Masked_Polar_Interp_clip.tif",
        "inputDataSet": "tdx_mad",
        "malardEnvironmentName": "DEVv2",
        "malardSyncURL": "http://localhost:9000",
        "malardAsyncURL": "ws://localhost:9000",
        "filters" : [{'column':'power','op':'gt','threshold':10000},{'column':'coh','op':'gt','threshold':0.6}, \
                     {'column':'demDiff','op':'lt','threshold':100}, {'column':'demDiffMadNew','op':'lt','threshold':10}, \
                     {'column':'demDiff','op':'gt','threshold':-100}]

    }



    def __init__(self, minX, maxX, minY, maxY, logFile=None):
        '''

        :param minX:
        :param maxX:
        :param minY:
        :param maxY:
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

        self.client = MalardClient(notebook=False)

        self.minX = minX
        self.maxX = maxX
        self.minY = minY
        self.maxY = maxY
        self.parentDsName = self.config('parentDsName')
        self.outputDataSet = self.config('outputDataSet')
        self.inputDataSet = self.config('inputDataSet')
        self.region = self.config('region')
        self.maskDataSet = self.config('maskDataSet')
        self.withinDataSets = self.config('withinDataSets')
        self.withinDataSetTypes = self.config('withinDataSetTypes')
        self.runName = self.config('runName')

        assert (self.maxX-self.minX) == (self.maxY-self.minY)
        self.size = maxX-minX
        self.dataSet = DataSet(parentDs=self.config('parentDsName'), dataSet=self.config('inputDataSet'), region=self.config('region'))





    def startProcess(self):
        self.logger.info('Starting gridcell: minX=%s, minY=%s, parentDs=%s, inputDataSet=%s, outputDataSet=%s, runName=%s,', self.minX, self.minY, self.parentDsName, self.inputDataSet, self.outputDataSet, self.runName)
        self.defineVariables()
        if os.path.exists(self.maskDataSetFile):
            self.data = self.filter()

            # To Geodata
            self.logger.info('Converting to Geodataset...')
            self.data = self.data.asGeoDataSet()
            self.applyMasks()

            # Calculate elevation difference
            if self.data.hasData():
                raster = RasterDataSet(self.config('referenceDem'))
                assert (self.maxX-self.minX)==(self.maxY-self.minY)
                buffer = (self.maxX-self.minX)*0.1
                self.data.calculateElevationDifference(raster, buffer=buffer)

                self.addStatistics()
                self.publish()
                self.logger.info("STATISTICS: %s", self.data.getStats())
        else:
            self.logger.info("No valid mask (fp=%s) found for %s, %s, %s, minX=%s, minY=%s, size=%s", self.maskDataSetFile, self.maskDataSet, 'Glacier', self.region, self.minX, self.minY, self.size)

        # shutdown
        self.logger.info("Finished process for: minX=%s, minY=%s, size=%s", self.minX, self.minY, self.size)
        self.logger.info('------------------------------------------------------------------')
        logging.shutdown()

        # clear variables
        sys.modules[__name__].__dict__.clear()


    def filter(self):
        filters = self.config('filters')
        self.logger.info("Filtering dataset=%s with criteria %s" % (self.inputDataSet, filters))
        result = self.client.executeQuery(self.dataSet,self.bbx,[],filters)
        self.logger.info("Result message: %s, %s" % (result.status, result.message))
        fp = result.resultFileName
        data = PointDataSet(fp, self.projection)
        # release cache of file
        self.client.releaseCacheHandle(result.resultFileName)
        data.addStatistic('%s_filtered' % self.inputDataSet, data.length())
        self.logger.info("Filter %s result count [%d]" % (self.inputDataSet, data.length()))
        return data

    def applyMasks(self):
        # Mask
        self.data.applyMask(self.maskDataSetFile,'Glacier')

        # Add column if point is inside masks
        for idx, i in enumerate(self.withinDataSetFiles):
            self.data.withinMask(i, self.withinDataSetTypes[idx])

    def addStatistics(self):
        self.logger.info('Adding additional statistics')
        # number of srtm and number of tandemX
        self.data.addStatistic('result_total', self.data.length())
        #stats['result_srtm'] = float(data.loc[data.dataset == 'SRTM', 'dataset'].count())
        #stats['result_tandemx'] = float(data.loc[data.dataset == 'TandemX', 'dataset'].count())
        self.data.addStatistic('result_avgX', self.data.mean('x'))
        self.data.addStatistic('result_avgY', self.data.mean('y'))
        self.data.addStatistic('result_offsetX', self.data.getStats()['result_avgX']-(self.minX+(self.size/2)))
        self.data.addStatistic('result_offsetY', self.data.getStats()['result_avgY']-(self.minX+(self.size/2)))

        # counts per year
        # @TODO do this in glacier years
        years=[x for x in range(self.minT.year, self.maxT.year+1)]
        for year in years:
            start = datetime.datetime(year,1,1,0,0)
            end = datetime.datetime(year+1,1,1,0,0)
            start = calendar.timegm(start.utctimetuple())
            end = calendar.timegm(end.utctimetuple())
            # count
            keyCount = "result_count_%s" % (year)
            peryear = float(self.data.data.loc[(self.data.data.time >= start) & (self.data.data.time <end)].shape[0])
            self.data.addStatistic(keyCount, peryear)
            # elevation difference
            elevDiff = "result_refDifference_%s" % (year)
            if peryear > 0.0:
                self.data.addStatistic(elevDiff, float(self.data.data.loc[(self.data.data.time >= start) & (self.data.data.time <end), 'refDifference'].mean()))
            else:
                self.data.addStatistic(elevDiff, 0.0)


    def publish(self, outEnvironment='/data/puma1/scratch/mtngla/ReadyData'):
        # get data as normal pandas dataframe without the geo ref
        data = self.data.getData(geo=False)

        outPath = os.path.join(outEnvironment, "ReadyData_%s_x%s_y%s.nc" % (self.minX, self.minY, self.size))
        xarr = data.to_xarray()
        xarr.to_netcdf(outPath)

        # publish
        self.logger.info('Publish new dataset...')
        result=self.query_async.publishGridCellPoints(self.parentDsName, self.outputDataSet, self.region, self.minX, self.minY, self.data.min('time'), self.size, outPath, self.projection)
        self.logger.info('Response: %s' %  result.json)
        # delete temporary file
        os.remove(outPath)

        # publish stats
        self.logger.info('Publish gridcell statistics...')
        response = self.query_sync.publishGridCellStats(self.parentDsName, self.runName, self.minX, self.minY, self.size, self.data.getStats())
        self.logger.info('Response: %s' % response)



    def defineVariables(self):
        self.query_sync = DataSetQuery(self.config('malardSyncURL'), self.config('malardEnvironmentName'))
        self.query_async = AsyncDataSetQuery(self.config('malardAsyncURL'), self.config('malardEnvironmentName'), False)
        # minT and maxT

        # get projection
        self.projection = self.client.getProjection(self.dataSet).proj4

        #minT and maxT
        bbx = self.client.boundingBox(self.dataSet)
        self.minT = bbx.minT
        self.maxT = bbx.maxT

        self.bbx = BoundingBox(self.minX, self.maxX, self.minY, self.maxY, self.minT, self.maxT)

        # masks
        mGla = self.query_sync.getGridCellMask(self.parentDsName, self.maskDataSet, 'Glacier', self.region, self.minX, self.minY, self.size)
        self.maskDataSetFile = json.loads(mGla)['fileName']

        self.withinDataSetFiles = []
        for i, el in enumerate(self.withinDataSets):
            if os.path.exists(el):
                self.withinDataSetFiles.append(el)
            else:
                mask = self.query_sync.getGridCellMask(self.parentDsName, el, self.withinDataSetTypes[i], self.region, self.minX, self.minY, self.size)
                self.withinDataSetFiles.append(json.loads(mask)['fileName'])

    @staticmethod
    def config(name):
        return MtnGlaGridcellProcess.__conf[name]
    def uncaughtErrorHandler(self, type, value, tb):
        self.logger.error("Uncaught exception", exc_info=(type, value, tb))

if __name__ ==  '__main__':
    #mtngla = MtnGlaGridcellProcess(400000, 500000, 0, 100000)
    #mtngla = MtnGlaGridcellProcess(500000, 600000, 0, 100000)
    #mtngla = MtnGlaGridcellProcess(700000, 800000, 0, 100000)
    #mtngla = MtnGlaGridcellProcess(500000, 600000, 100000, 200000)
    #mtngla = MtnGlaGridcellProcess(500000, 600000, -100000, 0)

    # error in this one for dem diff
    #mtngla = MtnGlaGridcellProcess(200000, 300000, -100000, 0)

    # mask file not found
    #mtngla = MtnGlaGridcellProcess(-200000, -100000, -200000, -100000)

    # alaska
    mtngla = MtnGlaGridcellProcess(-3900000, -3800000, -500000, -400000)

    # alaska out of bounds error
    #mtngla = MtnGlaGridcellProcess(-3300000, -3200000, 600000, 700000)

    #mtngla = MtnGlaGridcellProcess(1100000,1200000,400000,500000)


    mtngla.startProcess()
