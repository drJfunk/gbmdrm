from numpy import array, matrix, mean, zeros, where, arange, sqrt, arctan2,cos,sin,arccos

import numpy as np
import astropy.io.fits as fits




class GBMDRM(object):

    def __init__(self,drmfile,rspNum=-1):

        
        if rspNum >= 0:
            self.rspNum = rspNum+2
        else:
            self.rspNum="SPECRESP MATRIX"
            
        
        
        rspFile = fits.open(rspFileName)
        self.smeared = False
 
        self.fileName = rspFileName.split('/')[-1]
        

        self.chanData = rspFile['EBOUNDS'].data
      
        try:
            self.numEnergyBins = rspFile[self.rspNum].header['NUMEBINS']
        except(KeyError):
            self.numEnergyBins = rspFile[self.rspNum].header['NAXIS2']
        
        self.numDetChans = rspFile[self.rspNum].header['DETCHANS']
        try:
            self.det = rspFile["PRIMARY"].header['DETNAM']
        except(KeyError):
            self.det = rspFile["PRIMARY"].header['INSTRUME']
        
        
        self.photonE = array(zip([rspFile[self.rspNum].data["ENERG_LO"], rspFile[self.rspNum].data["ENERG_HI"]]))
        
        self.photonE = self.photonE.transpose()
        self.photonE = array(map(lambda x: x[0], self.photonE))
        self.chanWidth = self.photonE[:,1]-self.photonE[:,0]
        
        

        self.channelE = array(zip(rspFile["EBOUNDS"].data["E_MIN"], rspFile["EBOUNDS"].data["E_MAX"]))
        self.chanMin = self.channelE[:,0]
        self.chanMax = self.channelE[:,1]
        
        self.meanChan = array(map(mean,self.channelE))
        self.meanPhtE = array(map(mean,self.photonE))


        self._energySelection = where(array([True]*len(self.meanChan)))[0].tolist()

        
        #Main component of object
        self.drm = zeros((self.numEnergyBins, self.numDetChans))

        self._ConstructDRM(rspFile)
        
        self.binWidths = rspFile["EBOUNDS"].data["E_MAX"]-rspFile["EBOUNDS"].data["E_MIN"]
        self._CreatePhotonEvalEnergies()
        
        rspFile.close()


    def _CreatePhotonEvalEnergies(self):
        
        resFrac511 = 0.2 #lifted from BATSE!
        resExp = -0.15   #lifted from BATSE!

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
        """
        Set the function you would like to fold throught the RSP Matrx
        """
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
        """
        Returns the model counts folder through the RSP 
        matrix
        """



        
        self._ConvolveMatrix()
        if not ignore:
            modelCnts = self.counts[self._energySelection]
            
        else:
            modelCnts = self.counts
        return modelCnts
    
    def _ConvolveMatrix(self):
        
    
        
        
        self.counts = np.dot(self.drmTranspose,self.vec)
        
             
    def SetParams(self,*params):
        """
        Set the spectral parmaters of the model.
        """
        self.params = params
        self._EvalModel()

        
         
        
    ######################################################
    def _PoissonRate(self, meanCnts):

        numPhotons =  np.random.poisson(meanCnts,1)
        return numPhotons
        
    

    def SimSpectrum(self, *params):
        
        
        self.SetParams(params)
        meanCnts = self.GetModelCnts(ignore=True)
        
        
        simRates = array(map(self._PoissonRate,meanCnts))
            
        return simRates.T[0]
           
    
