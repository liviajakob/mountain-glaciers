import logging
import sys
from DataSets import *
from MalardClient.MalardClient import MalardClient
from MalardClient.DataSet import DataSet
from MalardClient.AsyncDataSetQuery import AsyncDataSetQuery
from MalardClient.DataSetQuery import DataSetQuery



class DemDiffMadProcess:



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
        "outputDataSet": "tdx4",
        "parentDsName": "mtngla",
        "region":"himalayas",
        "inputDataSet": "tdx2",
        "malardEnvironmentName": "DEVv2",
        "malardSyncURL": "http://localhost:9000",
        "malardAsyncURL": "ws://localhost:9000",
        "buffer": 15000,
        "maskDataSet": "RGIv60",
        "filters" : [{'column':'power','op':'gt','threshold':10000},{'column':'coh','op':'gt','threshold':0.6}]

    }



    def __init__(self, minX, maxX, minY, maxY, logFile=None, notebook=False):
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

        self.client = MalardClient(notebook=notebook)
        self.query_async = AsyncDataSetQuery(self.config('malardAsyncURL'), self.config('malardEnvironmentName'), False)



        self.minX = minX
        self.maxX = maxX
        self.minY = minY
        self.maxY = maxY
        assert (self.maxX-self.minX) == (self.maxY-self.minY)
        self.size = maxX-minX
        self.inputDataSet = DataSet(parentDs=self.config('parentDsName'), dataSet=self.config('inputDataSet'), region=self.config('region'))
        self.parentDsName = self.config('parentDsName')
        self.outputDataSet = self.config('outputDataSet')
        self.region = self.config('region')
        self.buffer = self.config('buffer')

        self.projection = self.client.getProjection(self.inputDataSet).proj4

        bbx = self.client.boundingBox(self.inputDataSet)
        self.minT = bbx.minT
        self.maxT = bbx.maxT


        # masks
        maskDataSet = self.config('maskDataSet')
        query_sync = DataSetQuery(self.config('malardSyncURL'), self.config('malardEnvironmentName'))
        mGla = query_sync.getGridCellMask(self.parentDsName, maskDataSet, 'Glacier', self.region, self.minX, self.minY, self.size)
        self.maskDataSetFile = json.loads(mGla)['fileName']





    def startProcess(self):
        self.logger.info('Starting gridcell: minX=%s, minY=%s, parentDs=%s, inputDataSet=%s, outputDataSet=%s', self.minX, self.minY, self.parentDsName, self.inputDataSet, self.outputDataSet)
        if os.path.exists(self.maskDataSetFile):
            self.data = self.filter()
            # Calculate elevation difference
            if self.data.hasData():

                # magic
                self.logger.info('Calculate demDiffMad...')
                self.data.data['demDiffMadNew'] = self.data.data['demDiff'].groupby([self.data.data['swathFileId'],self.data.data['wf_number']]).transform('mad')
                # delete gridcells outside cell (the ones that are within a buffer zone)
                self.logger.info('Cut down to gridcell...')
                filtered = self.data.data[((self.data.data.x>self.minX)&(self.data.data.x<self.maxX)&(self.data.data.y>self.minY)&(self.data.data.y<self.maxY))]
                self.logger.info('Count data before cut to gridcell =%s, after cut=%s', self.data.data.shape[0], filtered.shape[0])
                self.data.data = filtered
                if self.data.hasData():
                    self.publish()


            else:
                self.logger.info("No data in result query")

        else:
            self.logger.info("No glacier mask for area.")


        # shutdown
        self.logger.info("Finished process for: minX=%s, minY=%s, size=%s", self.minX, self.minY, self.size)
        self.logger.info('------------------------------------------------------------------')
        logging.shutdown()

        # clear variables
        sys.modules[__name__].__dict__.clear()


    def filter(self):
        filters = self.config('filters')
        self.logger.info("Filtering dataset=%s with criteria %s" % (self.inputDataSet.dataSet, filters))
        minXb = self.minX-self.buffer
        maxXb = self.maxX+self.buffer
        minYb = self.minY-self.buffer
        maxYb = self.maxY+self.buffer

        self.logger.info("Bounding box with buffer: minX=%s maxX=%s, minY=%s, mayY=%s" % (minXb, maxXb, minYb, maxYb))

        bbx_in = BoundingBox(minXb, maxXb, minYb, maxYb, self.minT, self.maxT)
        result = self.client.executeQuery(self.inputDataSet, bbx_in, projections=[], filters=filters)
        self.logger.info("Result message: %s, %s" % (result.status, result.message))
        data = PointDataSet(result.resultFileName, self.projection)

        self.logger.info("Data points count: %s" % (data.data.shape[0]))

        # release cache of file
        self.client.releaseCacheHandle(result.resultFileName)
        return data


    def publish(self, outEnvironment='/data/puma1/scratch/mtngla/ReadyData'):
        outPath = os.path.join(outEnvironment, "ReadyData_%s_x%s_y%s.nc" % (self.minX, self.minY, self.size))
        xarr = self.data.data.to_xarray()
        xarr.to_netcdf(outPath)

        # publish
        self.logger.info('Publish new dataset...')
        result=self.query_async.publishGridCellPoints(self.parentDsName, self.outputDataSet, self.region, self.minX, self.minY, self.data.min('time'), self.size, outPath, self.projection)
        self.logger.info('Response: %s' %  result.json)
        # delete temporary file
        os.remove(outPath)



    @staticmethod
    def config(name):
        return DemDiffMadProcess.__conf[name]
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
    #mtngla = MtnGlaGridcellProcess(-3900000, -3800000, -500000, -400000)

    # alaska out of bounds error
    #mtngla = MtnGlaGridcellProcess(-3300000, -3200000, 600000, 700000)

    #demDiffMadCalc = DemDiffMadProcess(1100000, 1200000, 400000, 500000)

    #minX=, minY=-400000,


    demDiffMadCalc = DemDiffMadProcess(400000, 500000, 0, 100000)
    #demDiffMadCalc = DemDiffMadProcess(-1800000, -1700000, -400000, -300000)
    demDiffMadCalc.startProcess()




