from numpy import array, matrix, mean, zeros, where, arange, sqrt, arctan2,cos,sin,arccos

import numpy as np
import astropy.io.fits as fits

import gbmpy


class SimRSP(object):

    def __init__(self,trigdat,det,mat_type=0,cspecfile=None,time=0.):
        self.mat_type=mat_type

        if det>12:
            #BGO
            self.in_edge= np.array([100.000,     105.579,    111.470,     117.689,     
	                           124.255,     131.188,
                                   138.507,     146.235,     154.394,     163.008,     172.103,     181.705,
                                   191.843,     202.546,     213.847,     225.778,     238.375,     251.675,
                                   265.716,     280.541,     296.194,     312.719,     330.167,     348.588,
                                   368.036,     388.570,     410.250,     433.139,     457.305,     482.820,
                                   509.757,     538.198,     568.226,     599.929,     633.401,     668.740,
                                   706.052,     745.444,     787.035,     830.946,     877.307,     926.255,
                                   977.933,     1032.49,     1090.10,     1150.92,     1215.13,     1282.93,
                                   1354.51,     1430.08,     1509.87,     1594.11,     1683.05,     1776.95,
                                   1876.09,     1980.77,     2091.28,     2207.96,     2331.15,     2461.21,
                                   2598.53,     2743.51,     2896.58,     3058.18,     3228.81,     3408.95,
                                   3599.15,     3799.96,     4011.97,     4235.81,     4472.14,     4721.65,
                                   4985.09,     5263.22,     5556.87,     5866.90,     6194.24,     6539.83,
                                   6904.71,     7289.95,     7696.67,     8126.09,     8579.47,     9058.15,
                                   9563.53,     10097.1,     10660.5,     11255.2,     11883.2,     12546.2,
                                   13246.2,     13985.2,     14765.5,     15589.3,     16459.1,     17377.4,
                                   18346.9,     19370.6,     20451.3,     21592.4,     22797.1,     24069.0,
                                   25411.8,     26829.7,     28326.6,     29907.0,     31575.6,     33337.3,
                                   35197.3,     37161.0,     39234.4,     41423.4,     43734.5,     46174.6,
                                   48750.8,     51470.7,     54342.5,     57374.4,     60575.5,     63955.2,
                                   67523.4,     71290.7,     75268.2,     79467.7,     83901.5,     88582.6,
                                   93524.9,     98742.9,     104252.,     110069.,     116210.,     122693.,
                                   129539.,     136766.,     144397.,     152453.,     160959.,     169939.,
                                   179421.,     189431.,     200000.],dtype=np.float32)

     
        else:
            self.in_edge=np.array([5.00000,     5.34000,     5.70312,     6.09094,     
	                 6.50513,     6.94748,
                         7.41991,     7.92447,     8.46333,     9.03884,     9.65349,     10.3099,
                         11.0110,     11.7598,     12.5594,     13.4135,     14.3256,     15.2997,
                         16.3401,     17.4513,     18.6380,     19.9054,     21.2589,     22.7045,
                         24.2485,     25.8974,     27.6584,     29.5392,     31.5479,     33.6931,
                         35.9843,     38.4312,     41.0446,     43.8356,     46.8164,     50.0000,
                         53.4000,     57.0312,     60.9094,     65.0513,     69.4748,     74.1991,
                         79.2446,     84.6333,     90.3884,     96.5349,     103.099,     110.110,
                         117.598,     125.594,     134.135,     143.256,     152.997,     163.401,
                         174.513,     186.380,     199.054,     212.589,     227.045,     242.485,
                         258.974,     276.584,     295.392,     315.479,     336.931,     359.843,
                         384.312,     410.446,     438.356,     468.164,     500.000,     534.000,
                         570.312,     609.094,     650.512,     694.748,     741.991,     792.446,
                         846.333,     903.884,     965.349,     1030.99,     1101.10,     1175.98,
                         1255.94,     1341.35,     1432.56,     1529.97,     1634.01,     1745.13,
                         1863.80,     1990.54,     2125.89,     2270.45,     2424.85,     2589.74,
                         2765.84,     2953.92,     3154.79,     3369.31,     3598.43,     3843.12,
                         4104.46,     4383.56,     4681.65,     5000.00,     5340.00,     5703.12,
                         6090.94,     6505.12,     6947.48,     7419.91,     7924.46,     8463.33,
                         9038.84,     9653.49,     10309.9,     11011.0,     11759.8,     12559.4,
                         13413.5,     14325.6,     15299.7,     16340.1,     17451.3,     18637.9,
                         19905.3,     21258.9,     22704.5,     24248.5,     25897.3,     27658.4,
                         29539.2,     31547.8,     33693.1,     35984.3,     38431.2,     41044.6,
                         43835.6,     46816.4,     50000.0],dtype=np.float32)
        



        
        # No this is fucking wrong!
        
        #self.out_edge = arange(1.,130.,dtype=np.float32)
        tmp = fits.open(cspecfile)
        
        self.out_edge = np.zeros(129,dtype=np.float32)

        self.out_edge[:-1] = tmp['EBOUNDS'].data['E_MIN']
        self.out_edge[-1]  = tmp['EBOUNDS'].data['E_MAX'][-1]
        
        self.numEnergyBins = 140
        
        
        self.numDetChans   = 128
        self.det = det
                
        self.photonE = np.array(zip(self.in_edge[:-1],self.in_edge[1:]))

        
        self.chanWidth = self.photonE[:,1]-self.photonE[:,0]
        
        

        self.channelE = np.array(zip(self.out_edge[:-1],self.out_edge[1:]))
        self.chanMin = self.channelE[:,0]
        self.chanMax = self.channelE[:,1]
        
        self.meanChan = array(map(mean,self.channelE))
        self.meanPhtE = array(map(mean,self.photonE))


        self._energySelection = where(array([True]*len(self.meanChan)))[0].tolist()




        # Space craft stuff from TRIGDAT
        trigdat   = fits.open(trigdat)
        trigtime  = trigdat['EVNTRATE'].header['TRIGTIME']
        tstart    = trigdat['EVNTRATE'].data['TIME']   - trigtime
        tstop     = trigdat['EVNTRATE'].data['ENDTIME'] - trigtime
        condition = np.logical_and(tstart<=time,time<=tstop)

        self.qauts = trigdat['EVNTRATE'].data['SCATTITD'][condition][0]
        self.sc_pos = trigdat['EVNTRATE'].data['EIC'][condition][0]
        trigdat.close()
        
        #self.binWidths = rspFile["EBOUNDS"].data["E_MAX"]-rspFile["EBOUNDS"].data["E_MIN"]






        self._CreatePhotonEvalEnergies()
        


    def SetLocation(self,ra,dec):

        self.ra =ra
        self.dec=dec
        az,el,geo_az,geo_el = self.scCoord(self.qauts,self.sc_pos,ra,dec)
        self.drm = gbmpy.gbmrsp(self.det,
                           az,
                           el,
                           geo_az,
                           geo_el,
                           self.in_edge,
                           self.out_edge,
                           self.mat_type)
        self.drmTranspose=self.drm.T

        
    def scCoord(self,sc_quat,sc_pos,ra,dec):
        scx=zeros(3)
        scy=zeros(3)
        scz=zeros(3)
        geodir=zeros(3)
        source_pos=zeros(3)
        source_pos_sc=zeros(3)
        


        scx[0] = (sc_quat[0]**2 - sc_quat[1]**2 - sc_quat[2]**2 + sc_quat[3]**2)
        scx[1] = 2.0 * (sc_quat[0] * sc_quat[1] + sc_quat[3] * sc_quat[2])
        scx[2] = 2.0 * (sc_quat[0] * sc_quat[2] - sc_quat[3] * sc_quat[1])
        scy[0] = 2.0 * (sc_quat[0] * sc_quat[1] - sc_quat[3] * sc_quat[2])
        scy[1] = (-sc_quat[0]**2 + sc_quat[1]**2 - sc_quat[2]**2 + sc_quat[3]**2)
        scy[2] = 2.0 * (sc_quat[1] * sc_quat[2] + sc_quat[3] * sc_quat[0])
        scz[0] = 2.0 * (sc_quat[0] * sc_quat[2] + sc_quat[3] * sc_quat[1])
        scz[1] = 2.0 * (sc_quat[1] * sc_quat[2] - sc_quat[3] * sc_quat[0])
        scz[2] = (-sc_quat[0]**2 - sc_quat[1]**2 + sc_quat[2]**2 + sc_quat[3]**2)




        geodir[0]= -scx.dot(sc_pos)
        geodir[1]= -scy.dot(sc_pos)
        geodir[2]= -scz.dot(sc_pos)


        denom = sqrt(geodir.dot(geodir))

        geodir/=denom

        geo_az = arctan2(geodir[1],geodir[0])

        if (geo_az < 0.0): 
            geo_az += 2 * np.pi
        while(geo_az>2*np.pi):
            geo_az-=2*np.pi

        geo_el = arctan2(sqrt(geodir[0]**2 + geodir[1]**2), geodir[2])


        dec=np.deg2rad(dec)
        ra =  np.deg2rad(ra)
        source_pos[0] = cos(dec) * cos(ra)
        source_pos[1] = cos(dec) * sin(ra)
        source_pos[2] = sin(dec)

        source_pos_sc[0]= scx.dot(source_pos)
        source_pos_sc[1]= scy.dot(source_pos)
        source_pos_sc[2]= scz.dot(source_pos)

        el = arccos(source_pos_sc[2])
        az = arctan2(source_pos_sc[1], source_pos_sc[0])

        if (az < 0.0): 
            az += 2 * np.pi
        el = 90-np.rad2deg(el)
        geo_el = 90 - np.rad2deg(geo_el)
        az = np.rad2deg(0.+az)
        geo_az = np.rad2deg(geo_az)
        return [az,el,geo_az,geo_el]

    def _CreatePhotonEvalEnergies(self):

        
        
        resFrac511 = 0.2 #lifted from RMFIT!
        resExp = -0.15   #lifted from RMFIT!

        binCenter = np.array(map(np.mean,self.photonE))
        

        resFrac = resFrac511*(binCenter/511.)**resExp
        resFWHM = binCenter*resFrac
        numEchans = np.ones(len(binCenter))

        self.lowEval = self.chanWidth<resFWHM/2.
        self.lowEvalWhere = where(self.lowEval)[0].tolist()
        
        self.medEval = self.chanWidth>=resFWHM/2.
        self.medEvalWhere = where(self.medEval)[0].tolist()
        
        numEchans[self.medEval]=3.
        self.highEval = self.chanWidth/2.>=resFWHM/3.
        self.highEvalWhere = where(self.highEval)[0].tolist()
        numEchans[self.highEval]=7.

        self.lowEne = binCenter[self.lowEval]
        self.medEne=np.array(map(lambda x,y: [x-0.333333*y,x,x+0.333333*y]
                                 ,binCenter[self.medEval],
                                 self.chanWidth[self.medEval]))
        self.highEne=np.array(map(lambda x,y: [x-0.5*y,
                                               x-0.333333*y,
                                               x-0.16667*y,
                                               x,
                                               x+0.16667*y,
                                               x+0.333333*y,x-0.5*y]
                                  ,binCenter[self.highEval]
                                  ,self.chanWidth[self.highEval]))
        
        
    def SetModel(self,model):
        self.model = model

    def _EvalModel(self):

        tmpCounts = zeros(len(self.photonE))

        lowRes = self.model(self.lowEne,*self.params)
        medRes = array(map(lambda x: sum(self.model(x,*self.params))/3.,self.medEne))



        hiRes =  array(map(lambda x: sum(self.model(x,*self.params))/7.,self.highEne))


        tmpCounts[self.lowEval]=lowRes
        tmpCounts[self.medEval]=medRes
        tmpCounts[self.highEval]=hiRes

        self.vec = tmpCounts*self.chanWidth
        

  

        
    
   
    def _GetChannel(self,energy):
        '''
        Private function that finds the channel for a given energy

        ____________________________________________________________
        arguments:
        energy: selection energy in keV

        '''

        if energy < self.chanMin[0]:
            return 0
        elif energy > self.chanMax[-1]:
            return len(self.chanMax)-1
    

        
        ch = 0
        for lo, hi in zip(self.chanMin,self.chanMax):

            if energy >= lo and energy <= hi:
                return ch
            else:
                ch+=1
  



    def SelectEnergies(self,selection):
        '''
        An array or nested array of energies (keV) is passed
        that specfies the energy selection(s) that will be used
        in the fitting procedure. This selection is memorized for
        plotting purposes later.


        '''

        # Make sure you've got an array
        selection = array(selection)
        self.selMins = []
        self.selMaxs = []

        # If the selection is a simple one
        if len(selection.shape) == 1:

                # Find out what the corresponding channels are
                tmp = map(self._GetChannel,selection)

                # Create a boolean array the length of all the channels
                tt = [False]*len(self.meanChan)

                # For all the good channels, flip the bits
                tt[tmp[0]:tmp[1]+1] = [True]*(tmp[1]-tmp[0]+1)

                # Record the max and min energies for plotting later
                self.emin = min(selection)
                self.emax = max(selection)

                # Record all the mins and maxes for plotting later
                self.selMins.append(self.chanMin[tmp[0]])
                self.selMaxs.append(self.chanMax[tmp[1]])

        # If instead we have a more complex selection...
        elif len(selection.shape) == 2:

                # Find the nested channel selections
                tmp = array(map(lambda x: [self._GetChannel(x[0]),self._GetChannel(x[1])] , selection))

                # Create a boolean array the length of all the channels
                tt = [False]*len(self.meanChan)

                # For each selection, flip the bits
                for x in tmp:

                    tt[x[0]:x[1]+1] = [True]*(x[1]+1-x[0])

                # Record all the good selection min and maxes
                self.selMins = self.chanMin[tmp[:,0]]
                self.selMaxs = self.chanMax[tmp[:,1]]

                self.emin = selection.min()
                self.emax = selection.max()

                    
        tt = array(tt)
        
        tt = where(tt)[0].tolist()


        # Save the energy selections
        self._energySelection = tt

    def GetEnergySelection(self):
        return self._energySelection
    
    def GetModelCnts(self,ignore=False):

        
        self._ConvolveMatrix()
        if not ignore:
            modelCnts = self.counts[self._energySelection]
            
        else:
            modelCnts = self.counts
        return modelCnts
    
    def _ConvolveMatrix(self):
        
    
        
        
        self.counts = np.dot(self.drmTranspose,self.vec)
        
             
    def SetParams(self,params):
        self.params = params
        self._EvalModel()

        
         
        
    ######################################################
    def _PoissonRate(self, meanCnts):

        numPhotons =  np.random.poisson(meanCnts,1)
        return numPhotons
        
    

    def SimSpectrum(self, params):
        
        
        self.SetParams(params)
        meanCnts = self.GetModelCnts(ignore=True)
        
        
        simRates = array(map(self._PoissonRate,meanCnts))
            
        return simRates.T[0]
           
    
