import datetime
import numpy as np
import pandas as pd
import sys
from csv import reader
import math
import gc

# TODO: add these properties from config.json

NaN = np.float32("NaN")
TFTConversion = True
CDSpecial = True
ScaleProperties = True
GenerateFutures = True
NumpredbasicperTime = 2
NumpredFuturedperTime = 2
PredictionsfromInputs = True
GarbageCollect = True
Dailyunit = 1

Tseq = 13

NumTimeSeriesCalculated = 0

startbold = "\033[1m"
resetfonts = "\033[0m"
startred = '\033[31m'

startpurple = '\033[35m'
startyellowbkg = '\033[43m'


def printexit(exitmessage):
    print(exitmessage)
    sys.exit()

def SetNewAverages(InputList): # name min max mean std

    results = np.empty(7, dtype = np.float32)
    results[0] = InputList[1]
    results[1] = InputList[2]
    results[2] = 1.0
    results[3] = InputList[3]
    results[4] = InputList[4]
    results[5] = InputList[3]
    results[6] = InputList[4]
    return results

def checkNaN(y):
    countNaN = 0
    countnotNaN = 0
    ctprt = 0
    if y is None:
        return
    if len(y.shape) == 2:
        for i in range(0, y.shape[0]):
            for j in range(0, y.shape[1]):
                if (np.math.isnan(y[i, j])):
                    countNaN += 1
                else:
                    countnotNaN += 1
    else:
        for i in range(0, y.shape[0]):
            for j in range(0, y.shape[1]):
                for k in range(0, y.shape[2]):
                    if (np.math.isnan(y[i, j, k])):
                        countNaN += 1
                        ctprt += 1
                        print(str(i) + ' ' + str(j) + ' ' + str(k))
                        if ctprt > 10:
                            sys.exit(0)
                    else:
                        countnotNaN += 1


def SetTakeroot(x, n):
    if np.isnan(x):
        return NaN
    if n == 3:
        return np.cbrt(x)
    elif n == 2:
        if x <= 0.0:
            return 0.0
        return np.sqrt(x)
    return x


def LinearLocationEncoding(TotalLoc):
    linear = np.empty(TotalLoc, dtype=float)
    for i in range(0, TotalLoc):
        linear[i] = float(i) / float(TotalLoc)
    return linear


def LinearTimeEncoding(Dateslisted):
    Firstdate = Dateslisted[0]
    numtofind = len(Dateslisted)
    dayrange = (Dateslisted[numtofind - 1] - Firstdate).days + 1
    linear = np.empty(numtofind, dtype=float)
    for i in range(0, numtofind):
        linear[i] = float((Dateslisted[i] - Firstdate).days) / float(dayrange)
    return linear


def P2TimeEncoding(numtofind):
    P2 = np.empty(numtofind, dtype=float)
    for i in range(0, numtofind):
        x = -1 + 2.0 * i / (numtofind - 1)
        P2[i] = 0.5 * (3 * x * x - 1)
    return P2


def P3TimeEncoding(numtofind):
    P3 = np.empty(numtofind, dtype=float)
    for i in range(0, numtofind):
        x = -1 + 2.0 * i / (numtofind - 1)
        P3[i] = 0.5 * (5 * x * x - 3) * x
    return P3


def P4TimeEncoding(numtofind):
    P4 = np.empty(numtofind, dtype=float)
    for i in range(0, numtofind):
        x = -1 + 2.0 * i / (numtofind - 1)
        P4[i] = 0.125 * (35 * x * x * x * x - 30 * x * x + 3)
    return P4


def WeeklyTimeEncoding(Dateslisted):
    numtofind = len(Dateslisted)
    costheta = np.empty(numtofind, dtype=float)
    sintheta = np.empty(numtofind, dtype=float)
    for i in range(0, numtofind):
        j = Dateslisted[i].date().weekday()
        theta = float(j) * 2.0 * math.pi / 7.0
        costheta[i] = math.cos(theta)
        sintheta[i] = math.sin(theta)
    return costheta, sintheta


def AnnualTimeEncoding(Dateslisted):
    numtofind = len(Dateslisted)
    costheta = np.empty(numtofind, dtype=float)
    sintheta = np.empty(numtofind, dtype=float)
    for i in range(0, numtofind):
        runningdate = Dateslisted[i]
        year = runningdate.year
        datebeginyear = datetime.datetime(year, 1, 1)
        displacement = (runningdate - datebeginyear).days
        daysinyear = (datetime.datetime(year, 12, 31) - datebeginyear).days + 1
        if displacement >= daysinyear:
            printexit("EXIT Bad Date ", runningdate)
        theta = float(displacement) * 2.0 * math.pi / float(daysinyear)
        costheta[i] = math.cos(theta)
        sintheta[i] = math.sin(theta)
    return costheta, sintheta


def ReturnEncoding(numtofind, Typeindex, Typevalue, LinearoverLocationEncoding, CosWeeklytimeEncoding,
                   SinWeeklytimeEncoding, CosAnnualtimeEncoding, SinAnnualtimeEncoding, LinearovertimeEncoding):
    Dummy = costheta = np.empty(0, dtype=float)
    if Typeindex == 1:
        return LinearoverLocationEncoding, Dummy, ('LinearSpace', 0., 1.0, 0.5, 0.2887), ('Dummy', 0., 0., 0., 0.)
    if Typeindex == 2:
        if Dailyunit == 1:
            return CosWeeklytimeEncoding, SinWeeklytimeEncoding, ('CosWeekly', -1.0, 1.0, 0., 0.7071), (
            'SinWeekly', -1.0, 1.0, 0., 0.7071)
        else:
            return Dummy, Dummy, ('Dummy', 0., 0., 0., 0.), ('Dummy', 0., 0., 0., 0.)
    if Typeindex == 3:
        return CosAnnualtimeEncoding, SinAnnualtimeEncoding, ('CosAnnual', -1.0, 1.0, 0., 0.7071), (
        'SinAnnual', -1.0, 1.0, 0., 0.7071)
    if Typeindex == 4:
        if Typevalue == 0:
            ConstArray = np.full(numtofind, 0.5, dtype=float)
            return ConstArray, Dummy, ('Constant', 0.5, 0.5, 0.5, 0.0), ('Dummy', 0., 0., 0., 0.)
        if Typevalue == 1:
            return LinearovertimeEncoding, Dummy, ('LinearTime', 0., 1.0, 0.5, 0.2887), ('Dummy', 0., 0., 0., 0.)
        if Typevalue == 2:
            return P2TimeEncoding(numtofind), Dummy, ('P2-Time', -1.0, 1.0, 0., 0.4472), ('Dummy', 0., 0., 0., 0.)
        if Typevalue == 3:
            return P3TimeEncoding(numtofind), Dummy, ('P3-Time', -1.0, 1.0, 0., 0.3780), ('Dummy', 0., 0., 0., 0.)
        if Typevalue == 4:
            return P4TimeEncoding(numtofind), Dummy, ('P4-Time', -1.0, 1.0, 0., 0.3333), ('Dummy', 0., 0., 0., 0.)
    if Typeindex == 5:
        costheta = np.empty(numtofind, dtype=float)
        sintheta = np.empty(numtofind, dtype=float)
        j = 0
        for i in range(0, numtofind):
            theta = float(j) * 2.0 * math.pi / Typevalue
            costheta[i] = math.cos(theta)
            sintheta[i] = math.sin(theta)
            j += 1
            if j >= Typevalue:
                j = 0
        return costheta, sintheta, ('Cos ' + str(Typevalue) + ' Len', -1.0, 1.0, 0., 0.7071), (
        'Sin ' + str(Typevalue) + ' Len', -1.0, 1.0, 0., 0.7071)


class Future:

    def __init__(self, name, daystart=0, days=[], wgt=1.0, classweight=1.0):
        self.name = name
        self.days = np.array(days)
        self.daystart = daystart
        self.wgts = np.full_like(self.days, wgt, dtype=float)
        self.size = len(self.days)
        self.classweight = classweight


def DynamicPropertyScaling(InputTimeSeries):
    Results = np.full(7, 0.0, dtype=np.float32)
    Results[1] = np.nanmax(InputTimeSeries, axis=(0, 1))
    Results[0] = np.nanmin(InputTimeSeries, axis=(0, 1))
    Results[3] = np.nanmean(InputTimeSeries, axis=(0, 1))
    Results[4] = np.nanstd(InputTimeSeries, axis=(0, 1))
    Results[2] = np.reciprocal(np.subtract(Results[1], Results[0]))
    Results[5] = np.multiply(Results[2], np.subtract(Results[3], Results[0]))
    Results[6] = np.multiply(Results[2], Results[4])
    return Results


class DataPreparation:

    def __init__(self, data_config, data_dir):

        self.config = data_config
        self.features = data_config['inputs_use']  # [(i, data_config['inputs'][i]) for i in data_config['inputs']]
        self.dir = data_dir

    def read_data(self, initial_date, end_date):

        GenerateSequences = True
        GenerateFutures = True  # Andrej bug
        NumpredFuturedperTime = 2
        NumpredbasicperTime = 2

        Dropyearlydata = 0
        InitialDate = initial_date + datetime.timedelta(
            days=Dropyearlydata)  # datetime.datetime(initial_date) + datetime.timedelta(days=Dropyearlydata)
        FinalDate = end_date  # datetime.datetime(end_date)
        NumberofTimeunits = (FinalDate - InitialDate).days + 1

        FeatureFiles = [self.config['input_files'][i] for i in self.config['input_files'] if
                        i in self.config['inputs_use']]
        FeatLen = len(self.config['inputs_use'])

        CasesFile = self.dir + '/' + self.config['targets']['Cases']
        DeathsFile = self.dir + '/' + self.config['targets']['Deaths']
        LocationdataFile = self.dir + '/' + self.config['support']['Population']
        LocationruralityFile = self.dir + '/' + self.config['support']['Rurality']
        VotingdataFile = self.dir + '/' + self.config['input_files']['Voting']
        AlaskaVotingdataFile = self.dir + '/' + self.config['input_files']['AlaskaVoting']

        Num_Time = NumberofTimeunits
        Nloc = self.config['support']['Nloc']
        NFIPS = self.config['support']['NFIPS']

        Locationfips = np.empty(self.config['support']['NFIPS'], dtype=int)
        Locationcolumns = []  # String version of FIPS
        FIPSintegerlookup = {}
        FIPSstringlookup = {}
        BasicInputTimeSeries = np.empty([Num_Time, Nloc, 2], dtype=np.float32)

        # Read in  cases Data into BasicInputTimeSeries
        with open(CasesFile, 'r') as read_obj:
            csv_reader = reader(read_obj)
            header = next(csv_reader)
            Ftype = header[0]
            if Ftype != 'FIPS':
                printexit('EXIT: Wrong file type Cases ' + Ftype)

            iloc = 0
            for nextrow in csv_reader:
                if len(nextrow) < NumberofTimeunits + 1 + Dropyearlydata:
                    printexit('EXIT: Incorrect row length Cases ' + str(iloc) + ' ' + str(len(nextrow)))
                # skip first entry
                localfips = nextrow[0]
                Locationcolumns.append(localfips)
                Locationfips[iloc] = int(localfips)
                FIPSintegerlookup[int(localfips)] = iloc
                FIPSstringlookup[localfips] = iloc
                for itime in range(0, NumberofTimeunits):
                    BasicInputTimeSeries[itime, iloc, 0] = nextrow[itime + 1 + Dropyearlydata]
                    if Dropyearlydata > 0:
                        floatlast = np.float(nextrow[Dropyearlydata])
                        BasicInputTimeSeries[itime, iloc, 0] = BasicInputTimeSeries[itime, iloc, 0] - floatlast
                iloc += 1
        if iloc != Nloc:
            printexit('EXIT Inconsistent location lengths Cases ' + str(iloc) + ' ' + str(Nloc))
        print('Read Cases data locations ' + str(Nloc) + ' Time Steps ' + str(Num_Time))

        # Read in deaths Data into BasicInputTimeSeries
        with open(DeathsFile, 'r') as read_obj:
            csv_reader = reader(read_obj)
            header = next(csv_reader)
            Ftype = header[0]
            if Ftype != 'FIPS':
                printexit('EXIT: Wrong file type Deaths ' + Ftype)

            iloc = 0
            for nextrow in csv_reader:
                if (len(nextrow) < NumberofTimeunits + 1 + Dropyearlydata):
                    printexit('EXIT: Incorrect row length Deaths ' + str(iloc) + ' ' + str(len(nextrow)))
                localfips = nextrow[0]
                if (Locationfips[iloc] != int(localfips)):
                    printexit('EXIT: Unexpected FIPS Deaths ' + localfips + ' ' + str(Locationfips[iloc]))
                for itime in range(0, NumberofTimeunits):
                    BasicInputTimeSeries[itime, iloc, 1] = nextrow[itime + 1 + Dropyearlydata]
                    if Dropyearlydata > 0:
                        floatlast = np.float(nextrow[Dropyearlydata])
                        BasicInputTimeSeries[itime, iloc, 1] = BasicInputTimeSeries[itime, iloc, 1] - floatlast
                iloc += 1
        # End Reading in deaths data

        if iloc != Nloc:
            printexit('EXIT Inconsistent location lengths ' + str(iloc) + ' ' + str(Nloc))
        print('Read Deaths data locations ' + str(Nloc) + ' Time Steps ' + str(Num_Time))

        Locationname = ['Empty'] * NFIPS
        Locationstate = ['Empty'] * NFIPS
        Locationpopulation = np.empty(NFIPS, dtype=int)
        with open(LocationdataFile, 'r', encoding='latin1') as read_obj:
            csv_reader = reader(read_obj)
            header = next(csv_reader)
            Ftype = header[0]
            if Ftype != 'FIPS':
                printexit('EXIT: Wrong file type Prop Data ' + Ftype)

            iloc = 0
            for nextrow in csv_reader:
                localfips = int(nextrow[0])
                if localfips in FIPSintegerlookup.keys():
                    jloc = FIPSintegerlookup[localfips]
                    Locationname[jloc] = nextrow[4]
                    Locationstate[jloc] = nextrow[3]
                    Locationpopulation[jloc] = int(nextrow[2])
                    iloc += 1  # just counting lines
                else:
                    printexit('EXIT Inconsistent FIPS ' + str(iloc) + ' ' + str(localfips))
                    # END setting NFIPS location properties

        DemVoting = np.full(NFIPS, -1.0, dtype=np.float32)
        RepVoting = np.full(NFIPS, -1.0, dtype=np.float32)
        with open(VotingdataFile, 'r', encoding='latin1') as read_obj:
            csv_reader = reader(read_obj)
            header = next(csv_reader)
            Ftype = header[0]
            if Ftype != 'state_name':
                printexit('EXIT: Wrong file type Voting Data ' + Ftype)

            iloc = 0
            for nextrow in csv_reader:
                localfips = int(nextrow[1])
                if localfips > 2900 and localfips < 2941:  # Alaska not useful
                    continue
                if localfips in FIPSintegerlookup.keys():
                    jloc = FIPSintegerlookup[localfips]
                    if DemVoting[jloc] >= 0.0:
                        printexit('EXIT Double Setting of FIPS ' + str(iloc) + ' ' + str(localfips))
                    DemVoting[jloc] = nextrow[8]
                    RepVoting[jloc] = nextrow[7]
                    iloc += 1  # just counting lines
                else:
                    printexit('EXIT Inconsistent FIPS ' + str(iloc) + ' ' + str(localfips))

        with open(AlaskaVotingdataFile, 'r', encoding='utf-8-sig') as read_obj:  # remove ufeff
            csv_reader = reader(read_obj)
            header = next(csv_reader)
            Ftype = header[0]
            if Ftype != 'SpecialAlaska':
                printexit('EXIT: Wrong file type Alaska Voting Data ' + Ftype)

            iloc = 0
            for nextrow in csv_reader:
                localfips = int(nextrow[1])
                if localfips in FIPSintegerlookup.keys():
                    jloc = FIPSintegerlookup[localfips]
                    if DemVoting[jloc] >= 0.0:
                        printexit('EXIT Double Setting of FIPS ' + str(iloc) + ' ' + str(localfips))
                    DemVoting[jloc] = float(nextrow[2]) * 42.77 / 36.5
                    RepVoting[jloc] = float(nextrow[3]) * 52.83 / 51.3
                    iloc += 1  # just counting lines
                else:
                    printexit('EXIT Inconsistent FIPS ' + str(iloc) + ' ' + str(localfips))

        for iloc in range(0, NFIPS):
            if DemVoting[iloc] >= 0.0:
                continue
            print(str(iloc) + ' Missing Votes ' + str(Locationfips[iloc]) + ' ' + Locationname[iloc] + ' ' +
                  Locationstate[iloc] + ' pop ' + str(Locationpopulation[iloc]))
            DemVoting[iloc] = 0.5
            RepVoting[iloc] = 0.5

        # Set Static Properties of the Nloc studied locations
        # Order is Static, Dynamic, Cases, Deaths
        # Voting added as 13th covariate
        NpropperTimeDynamic = FeatLen
        NpropperTimeStatic = 0

        NpropperTime = NpropperTimeStatic + NpropperTimeDynamic + 2
        InputPropertyNames = [] * NpropperTime
        Property_is_Intensive = np.full(NpropperTime, True, dtype=np.bool)

        RuralityFile = self.dir + '/' + 'Rurality.csv'
        RuralityRange = self.config['support']['RuralityRange']
        RuralityCut = True

        Locationrurality = np.empty(NFIPS)
        Locationmad = np.empty(NFIPS)

        skips = []

        with open(RuralityFile, 'r') as read_obj:
            csv_reader = reader(read_obj)
            header = next(csv_reader)
            Ftype = header[0]
            if Ftype != 'FIPS':
                printexit('EXIT: Wrong file type Prop Data ' + Ftype)

            iloc = 0
            for row in csv_reader:
                localfips = int(row[0])
                if localfips in FIPSintegerlookup.keys():
                    jloc = FIPSintegerlookup[localfips]
                    Locationrurality[jloc] = row[1]
                    Locationmad[jloc] = row[2]
                    iloc += 1
                else:
                    print('FIPS not in lookup table ' + str(iloc) + ' ' + str(localfips))
                    skips.append(iloc)

        Locationrurality[81] = None
        Locationrurality[2412] = None
        Locationmad[81] = None
        Locationmad[2412] = None

        Propfilenames = FeatureFiles
        Propnames = [i[:-4] for i in Propfilenames]

        # TODO add voting conditions

        NIHDATADIR = self.dir + '/'
        numberfiles = len(Propnames)
        NpropperTimeStatic = 0
        if NpropperTimeDynamic != numberfiles:
            printexit('EXIT: Dynamic Properties set wrong ' + str(numberfiles) + ' ' + str(NpropperTimeDynamic))
        DynamicPropertyTimeSeries = np.empty([Num_Time, Nloc, numberfiles], dtype=np.float32)
        enddifference = NaN

        for ifiles in range(0, numberfiles):
            InputPropertyNames.append(Propnames[ifiles])
            if Propfilenames[ifiles] == 'NOFILE':  # Special case of Voting Data
                for iloc in range(0, Nloc):
                    Demsize = DemVoting[iloc]
                    RepSize = RepVoting[iloc]
                    Votingcovariate = Demsize / (RepSize + Demsize)
                    DynamicPropertyTimeSeries[:, iloc, ifiles] = Votingcovariate
                continue  # over ifile loop

            DynamicPropFile = NIHDATADIR + Propfilenames[ifiles]

            # Read in  Covariate Data into DynamicPropertyTimeSeries
            print(DynamicPropFile)
            with open(DynamicPropFile, 'r') as read_obj:
                csv_reader = reader(read_obj)
                header = next(csv_reader)
                skip = 2
                Ftype = header[0]
                if Ftype != 'Name':
                    printexit('EXIT: Wrong file type ' + Ftype)
                Ftype = header[skip - 1]
                if Ftype != 'FIPS':
                    printexit('EXIT: Wrong file type ' + Ftype)
                # Check Date
                hformat = '%Y-%m-%d'
                firstdate = datetime.datetime.strptime(header[skip], hformat)
                tdelta = (firstdate - InitialDate).days
                if tdelta > 0:
                    printexit('Missing Covariate Data start -- adjust Dropearlydata ' + str(tdelta))
                lastdate = datetime.datetime.strptime(header[len(header) - 1], hformat)
                enddifference1 = (FinalDate - lastdate).days
                if math.isnan(enddifference):
                    enddifference = enddifference1
                    print('Missing days at the end ' + str(enddifference))
                else:
                    if enddifference != enddifference1:
                        printexit('EXIT: Incorrect time length ' + Propnames[ifiles] + ' expected ' + str(
                            enddifference) + ' actual ' + str(enddifference1))
                iloc = 0

                # Test
                if lastdate > FinalDate:
                    tempInt = NumberofTimeunits
                else:
                    tempInt = NumberofTimeunits - enddifference

                for nextrow in csv_reader:
                 #   if iloc == 0:
                 #       print(len(nextrow) != NumberofTimeunits + skip - enddifference - tdelta)
                    if len(nextrow) != NumberofTimeunits + skip - enddifference - tdelta:
                        printexit(
                            'EXIT: Incorrect row length ' + Propnames[ifiles] + ' Location ' + str(iloc) + ' ' + str(
                                len(nextrow)))
                    localfips = nextrow[skip - 1]
                    jloc = FIPSstringlookup[localfips]
                    for itime in range(0, tempInt):  # NumberofTimeunits - enddifference
                        DynamicPropertyTimeSeries[itime, jloc, ifiles] = nextrow[itime + skip - tdelta]
                    # Use previous week value for missing data at the end
                    if tempInt != NumberofTimeunits:
                        for itime in range(tempInt, NumberofTimeunits):  # NumberofTimeunits - enddifference
                            DynamicPropertyTimeSeries[itime, jloc, ifiles] = DynamicPropertyTimeSeries[
                                itime - 7, jloc, ifiles]
                    iloc += 1
            # End Reading in dynamic property data

            if iloc != Nloc:
                printexit('EXIT Inconsistent location lengths ' + Propnames[ifiles] + str(iloc) + ' ' + str(Nloc))
            print('Read ' + Propnames[ifiles] + ' data for locations ' + str(Nloc) + ' Time Steps ' + str(
                Num_Time) + ' Days dropped at start ' + str(-tdelta))
        if RuralityCut:
            uselocation = np.full(Nloc, False, dtype=np.bool)
            if len(RuralityRange) > 0:
                for jloc in range(0, Nloc):
                    if Locationrurality[jloc] >= RuralityRange[0] and Locationrurality[jloc] <= RuralityRange[1]:
                        uselocation[jloc] = True
                TotRuralityCut = uselocation.sum()
                NumberCut = Nloc - TotRuralityCut
                print(
                    ' Rurality Cut ' + str(RuralityRange) + ' removes ' + str(Nloc - TotRuralityCut) + ' of ' + str(
                        Nloc))

            else:
                printexit('EXIT There are no rurality criteria')

            if TotRuralityCut > 0:
                NewNloc = Nloc - NumberCut
                NewNFIPS = NewNloc
                NewLocationfips = np.empty(NewNFIPS, dtype=int)  # integer version of FIPs
                NewLocationcolumns = []  # String version of FIPS
                NewFIPSintegerlookup = {}
                NewFIPSstringlookup = {}
                NewBasicInputTimeSeries = np.empty([Num_Time, NewNloc, 2], dtype=np.float32)
                NewLocationname = ['Empty'] * NewNFIPS
                NewLocationstate = ['Empty'] * NewNFIPS
                NewLocationpopulation = np.empty(NewNFIPS, dtype=int)
                NewLocationrurality = np.empty(NewNFIPS)
                NewLocationmad = np.empty(NewNFIPS)
                NewDynamicPropertyTimeSeries = np.empty([Num_Time, NewNloc, numberfiles], dtype=np.float32)

                Newiloc = 0
                for iloc in range(0, Nloc):
                    if not uselocation[iloc]:
                        continue
                    NewBasicInputTimeSeries[:, Newiloc, :] = BasicInputTimeSeries[:, iloc, :]
                    NewDynamicPropertyTimeSeries[:, Newiloc, :] = DynamicPropertyTimeSeries[:, iloc, :]
                    localfips = Locationcolumns[iloc]
                    NewLocationcolumns.append(localfips)
                    NewLocationfips[Newiloc] = int(localfips)
                    NewFIPSintegerlookup[int(localfips)] = Newiloc
                    NewFIPSstringlookup[localfips] = Newiloc
                    NewLocationpopulation[Newiloc] = Locationpopulation[iloc]
                    NewLocationstate[Newiloc] = Locationstate[iloc]
                    NewLocationname[Newiloc] = Locationname[iloc]
                    NewLocationrurality[Newiloc] = Locationrurality[iloc]
                    NewLocationmad[Newiloc] = Locationmad[iloc]
                    Newiloc += 1

                BasicInputTimeSeries = NewBasicInputTimeSeries
                DynamicPropertyTimeSeries = NewDynamicPropertyTimeSeries
                Locationname = NewLocationname
                Locationstate = NewLocationstate
                Locationpopulation = NewLocationpopulation
                FIPSstringlookup = NewFIPSstringlookup
                FIPSintegerlookup = NewFIPSintegerlookup
                Locationcolumns = NewLocationcolumns
                Locationfips = NewLocationfips
                Locationrurality = NewLocationrurality
                Locationmad = NewLocationmad
                NFIPS = NewNFIPS
                Nloc = NewNloc

        MADRange = self.config['support']['MADRange']

        if RuralityCut:
            uselocation = np.full(Nloc, False, dtype=np.bool)
            if len(MADRange) > 0:
                for jloc in range(0, Nloc):
                    if (Locationmad[jloc] >= MADRange[0]) & (Locationmad[jloc] < MADRange[1]):
                        uselocation[jloc] = True
                TotMADCut = uselocation.sum()
                NumberCut = Nloc - TotMADCut
                print(' MAD Cut ' + str(MADRange) + ' removes ' + str(Nloc - TotMADCut) + ' of ' + str(Nloc))

            else:
                printexit('EXIT There are no rurality criteria')

            if TotRuralityCut > 0:
                NewNloc = Nloc - NumberCut
                NewNFIPS = NewNloc
                NewLocationfips = np.empty(NewNFIPS, dtype=int)  # integer version of FIPs
                NewLocationcolumns = []  # String version of FIPS
                NewFIPSintegerlookup = {}
                NewFIPSstringlookup = {}
                NewBasicInputTimeSeries = np.empty([Num_Time, NewNloc, 2], dtype=np.float32)
                NewLocationname = ['Empty'] * NewNFIPS
                NewLocationstate = ['Empty'] * NewNFIPS
                NewLocationpopulation = np.empty(NewNFIPS, dtype=int)
                NewDynamicPropertyTimeSeries = np.empty([Num_Time, NewNloc, numberfiles], dtype=np.float32)

                Newiloc = 0
                for iloc in range(0, Nloc):
                    if not uselocation[iloc]:
                        continue
                    NewBasicInputTimeSeries[:, Newiloc, :] = BasicInputTimeSeries[:, iloc, :]
                    NewDynamicPropertyTimeSeries[:, Newiloc, :] = DynamicPropertyTimeSeries[:, iloc, :]
                    localfips = Locationcolumns[iloc]
                    NewLocationcolumns.append(localfips)
                    NewLocationfips[Newiloc] = int(localfips)
                    NewFIPSintegerlookup[int(localfips)] = Newiloc
                    NewFIPSstringlookup[localfips] = Newiloc
                    NewLocationpopulation[Newiloc] = Locationpopulation[iloc]
                    NewLocationstate[Newiloc] = Locationstate[iloc]
                    NewLocationname[Newiloc] = Locationname[iloc]
                    Newiloc += 1

                BasicInputTimeSeries = NewBasicInputTimeSeries
                DynamicPropertyTimeSeries = NewDynamicPropertyTimeSeries
                Locationname = NewLocationname
                Locationstate = NewLocationstate
                Locationpopulation = NewLocationpopulation
                FIPSstringlookup = NewFIPSstringlookup
                FIPSintegerlookup = NewFIPSintegerlookup
                Locationcolumns = NewLocationcolumns
                Locationfips = NewLocationfips
                NFIPS = NewNFIPS
                Nloc = NewNloc

        # Remove special status of Cases and Deaths
        if CDSpecial:

            NewNpropperTimeDynamic = NpropperTimeDynamic + 2
            NewNpropperTime = NpropperTimeStatic + NewNpropperTimeDynamic

            NewProperty_is_Intensive = np.full(NewNpropperTime, True, dtype=np.bool)
            NewInputPropertyNames = []
            NewDynamicPropertyTimeSeries = np.empty([Num_Time, NewNloc, NewNpropperTimeDynamic], dtype=np.float32)

            for casesdeaths in range(0, 2):
                NewDynamicPropertyTimeSeries[:, :, casesdeaths] = BasicInputTimeSeries[:, :, casesdeaths]
            BasicInputTimeSeries = None

            for iprop in range(0, NpropperTimeStatic):
                NewInputPropertyNames.append(InputPropertyNames[iprop])
                NewProperty_is_Intensive[iprop] = Property_is_Intensive[iprop]
            NewProperty_is_Intensive[NpropperTimeStatic] = False
            NewProperty_is_Intensive[NpropperTimeStatic + 1] = False
            NewInputPropertyNames.append('Cases')
            NewInputPropertyNames.append('Deaths')
            for ipropdynamic in range(0, NpropperTimeDynamic):
                Newiprop = NpropperTimeStatic + 2 + ipropdynamic
                iprop = NpropperTimeStatic + ipropdynamic
                NewDynamicPropertyTimeSeries[:, :, Newiprop] = DynamicPropertyTimeSeries[:, :, iprop]
                NewInputPropertyNames.append(InputPropertyNames[iprop])
                NewProperty_is_Intensive[Newiprop] = Property_is_Intensive[iprop]

            NpropperTimeDynamic = NewNpropperTimeDynamic
            NpropperTime = NewNpropperTime
            DynamicPropertyTimeSeries = NewDynamicPropertyTimeSeries
            InputPropertyNames = NewInputPropertyNames
            Property_is_Intensive = NewProperty_is_Intensive

        NpropperTimeMAX = NpropperTime

        if ScaleProperties:
            QuantityTakeroot = np.full(NpropperTimeMAX, 1, dtype=np.int)

            if CDSpecial:
                QuantityTakeroot[NpropperTimeStatic] = 2
                QuantityTakeroot[NpropperTimeStatic + 1] = 2

            for iprop in range(0, NpropperTimeMAX):
                if QuantityTakeroot[iprop] >= 2:
                    if iprop < NpropperTime:
                        for itime in range(0, NumberofTimeunits):
                            for iloc in range(0, Nloc):
                                DynamicPropertyTimeSeries[itime, iloc, iprop - NpropperTimeStatic] = SetTakeroot(
                                    DynamicPropertyTimeSeries[itime, iloc, iprop - NpropperTimeStatic],
                                    QuantityTakeroot[iprop])
                    # else:
                    #     for itime in range(0, NumberofTimeunits):
                    #         for iloc in range(0, Nloc):
                    #             CalculatedTimeSeries[itime, iloc, iprop - NpropperTime] = SetTakeroot(
                    #                 CalculatedTimeSeries[itime, iloc, iprop - NpropperTime], QuantityTakeroot[iprop])

            QuantityStatisticsNames = ['Min', 'Max', 'Norm', 'Mean', 'Std', 'Normed Mean', 'Normed Std']
            QuantityStatistics = np.zeros([NpropperTimeMAX, 7], dtype=np.float32)

            if (NpropperTimeDynamic > 0) or (NumTimeSeriesCalculated > 0):
                for iprop in range(NpropperTimeStatic, NpropperTimeStatic + NpropperTimeDynamic):
                    QuantityStatistics[iprop, :] = DynamicPropertyScaling(
                        DynamicPropertyTimeSeries[:, :, iprop - NpropperTimeStatic])

                NormedDynamicPropertyTimeSeries = np.empty_like(DynamicPropertyTimeSeries)
                for iprop in range(NpropperTimeStatic, NpropperTimeStatic + NpropperTimeDynamic):
                    NormedDynamicPropertyTimeSeries[:, :, iprop - NpropperTimeStatic] = np.multiply(
                        (DynamicPropertyTimeSeries[:, :, iprop - NpropperTimeStatic]
                         - QuantityStatistics[iprop, 0]), QuantityStatistics[iprop, 2])

                BasicInputStaticProps = None
                DynamicPropertyTimeSeries = None
                print(startbold + "Properties scaled" + resetfonts)

            line = 'Name   '
            for propval in range(0, 7):
                line += QuantityStatisticsNames[propval] + '    '
            print('\n' + startbold + startpurple + line + resetfonts)
            for iprop in range(0, NpropperTimeMAX):
                if iprop == NpropperTimeStatic:
                    print('\n')
                line = startbold + startpurple + str(iprop) + ' ' + InputPropertyNames[
                    iprop] + resetfonts + ' Root ' + str(QuantityTakeroot[iprop])
                for propval in range(0, 7):
                    line += ' ' + str(round(QuantityStatistics[iprop, propval], 3))
                print(line)

        LengthFutures = 0
        GenerateFutures = True  # Andrej bug
        NumpredFuturedperTime = 2
        NumpredbasicperTime = 2
        if GenerateFutures:  # daystart overwritten
            Secondday = Future('day2', daystart=23, days=[2], classweight=1. / 14.)
            Thirdday = Future('day3', daystart=24, days=[3], classweight=1. / 14.)
            Fourthday = Future('day4', daystart=25, days=[4], classweight=1. / 14.)
            Fifthday = Future('day5', daystart=26, days=[5], classweight=1. / 14.)
            Sixthday = Future('day6', daystart=27, days=[6], classweight=1. / 14.)
            Seventhday = Future('day7', daystart=27, days=[7], classweight=1. / 14.)
            day8 = Future('day8', daystart=28, days=[8], classweight=1. / 14.)
            day9 = Future('day9', daystart=29, days=[9], classweight=1. / 14.)
            day10 = Future('day10', daystart=30, days=[10], classweight=1. / 14.)
            day11 = Future('day11', daystart=31, days=[11], classweight=1. / 14.)
            day12 = Future('day12', daystart=32, days=[12], classweight=1. / 14.)
            day13 = Future('day13', daystart=33, days=[13], classweight=1. / 14.)
            day14 = Future('day14', daystart=34, days=[14], classweight=1. / 14.)
            day15 = Future('day15', daystart=35, days=[15], classweight=1. / 14.)

            Futures = [Secondday, Thirdday, Fourthday, Fifthday, Sixthday, Seventhday, day8, day9, day10, day11, day12,
                       day13, day14, day15]
            Futures = []
            for ifuture in range(0, 14):
                xx = Future(str(ifuture + 1), days=[ifuture + 2])
                Futures.append(xx)
            LengthFutures = len(Futures)
            Futuresmaxday = 0
            Futuresmaxweek = 0
            for i in range(0, LengthFutures):
                j = len(Futures[i].days)
                if j == 1:
                    Futuresmaxday = max(Futuresmaxday, Futures[i].days[0])
                else:
                    Futuresmaxweek = max(Futuresmaxweek, Futures[i].days[j - 1])
                Futures[i].daystart -= Dropyearlydata
                if Futures[i].daystart < 0: Futures[i].daystart = 0

        OriginalNloc = Nloc

        MappedLocations = np.arange(0, Nloc, dtype=np.int)
        LookupLocations = np.arange(0, Nloc, dtype=np.int)
        MappedNloc = Nloc

        InputSource = ['Dynamic'] * 14
        InputSourceNumber = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

        PredSource = ['Dynamic', 'Dynamic']
        PredSourceNumber = [0, 1]
        FuturedPred = [1, 1]

        # Encodings
        PropTypes = ['Spatial', 'TopDown', 'TopDown', 'TopDown', 'TopDown', 'TopDown', 'Weekly']
        PropValues = [0, 0, 1, 2, 3, 4, 0]

        PredTypes = Types = ['Spatial', 'TopDown', 'TopDown', 'TopDown', 'TopDown', 'TopDown', 'Weekly']
        PredValues = [0, 0, 1, 2, 3, 4, 0]

        PredTypes = []
        PredValues = []

        if len(InputSource) != len(InputSourceNumber):
            printexit(' Inconsistent Source Lengths ' + str(len(InputSource)) + str(len(InputSourceNumber)))
        if len(PredSource) != len(PredSourceNumber):
            printexit(' Inconsistent Rediction Lengths ' + str(len(PredSource)) + str(len(PredSourceNumber)))

        if len(PredSource) > 0:  # set up Predictions
            NumpredbasicperTime = len(PredSource)
            FuturedPointer = np.full(NumpredbasicperTime, -1, dtype=np.int)
            NumpredFuturedperTime = 0
            NumpredfromInputsperTime = 0
            for ipred in range(0, len(PredSource)):
                if PredSource[ipred] == 'Dynamic':
                    NumpredfromInputsperTime += 1
            countinputs = 0
            countcalcs = 0
            for ipred in range(0, len(PredSource)):
                if not (PredSource[ipred] == 'Dynamic' or PredSource[ipred] == 'Calc'):
                    printexit('Illegal Prediction ' + str(ipred) + ' ' + PredSource[ipred])
                if PredSource[ipred] == 'Dynamic':
                    countinputs += 1
                else:
                    countcalcs += 1
                if FuturedPred[ipred] >= 0:
                    if LengthFutures > 0:
                        FuturedPred[ipred] = NumpredFuturedperTime
                        FuturedPointer[ipred] = NumpredFuturedperTime
                        NumpredFuturedperTime += 1
                    else:
                        FuturedPred[ipred] = -1

        else:  # Set defaults
            NumpredfromInputsperTime = NumpredFuturedperTime
            FuturedPointer = np.full(NumpredbasicperTime, -1, dtype=np.int)
            PredSource = []
            PredSourceNumber = []
            FuturedPred = []
            futurepos = 0
            for ipred in range(0, NumpredFuturedperTime):
                PredSource.append('Dynamic')
                PredSourceNumber.append(ipred)
                futured = -1
                if LengthFutures > 0:
                    futured = futurepos
                    FuturedPointer[ipred] = futurepos
                    futurepos += 1
                FuturedPred.append(futured)
            for ipred in range(0, NumTimeSeriesCalculated):
                PredSource.append('Calc')
                PredSourceNumber.append(ipred)
                FuturedPred.append(-1)
            print('Number of Predictions ' + str(len(PredSource)))

        PropertyNameIndex = np.empty(NpropperTime, dtype=np.int32)
        PropertyAverageValuesPointer = np.empty(NpropperTime, dtype=np.int32)
        for iprop in range(0, NpropperTime):
            PropertyNameIndex[iprop] = iprop  # names
            PropertyAverageValuesPointer[iprop] = iprop  # normalizations

        # Reset Source -- if OK as read don't set InputSource InputSourceNumber
        # Reset NormedDynamicPropertyTimeSeries and NormedInputStaticProps
        # Reset NpropperTime = NpropperTimeStatic + NpropperTimeDynamic
        if len(InputSource) > 0:  # Reset Input Source
            NewNpropperTimeStatic = 0
            NewNpropperTimeDynamic = 0
            for isource in range(0, len(InputSource)):
                if InputSource[isource] == 'Static':
                    NewNpropperTimeStatic += 1
                if InputSource[isource] == 'Dynamic':
                    NewNpropperTimeDynamic += 1
            NewNormedDynamicPropertyTimeSeries = np.empty([Num_Time, Nloc, NewNpropperTimeDynamic], dtype=np.float32)
            NewNormedInputStaticProps = np.empty([Nloc, NewNpropperTimeStatic], dtype=np.float32)
            NewNpropperTime = NewNpropperTimeStatic + NewNpropperTimeDynamic
            NewPropertyNameIndex = np.empty(NewNpropperTime, dtype=np.int32)
            NewPropertyAverageValuesPointer = np.empty(NewNpropperTime, dtype=np.int32)
            countstatic = 0
            countdynamic = 0
            for isource in range(0, len(InputSource)):
                if InputSource[isource] == 'Dynamic':
                    OlddynamicNumber = InputSourceNumber[isource]
                    NewNormedDynamicPropertyTimeSeries[:, :, countdynamic] = NormedDynamicPropertyTimeSeries[:, :,
                                                                             OlddynamicNumber]
                    NewPropertyNameIndex[countdynamic + NewNpropperTimeStatic] = PropertyNameIndex[
                        OlddynamicNumber + NpropperTimeStatic]
                    NewPropertyAverageValuesPointer[countdynamic + NewNpropperTimeStatic] = \
                        PropertyAverageValuesPointer[OlddynamicNumber + NpropperTimeStatic]
                    countdynamic += 1

                else:
                    printexit('Illegal Property ' + str(isource) + ' ' + InputSource[isource])

        else:  # pretend data altered
            NewPropertyNameIndex = PropertyNameIndex
            NewPropertyAverageValuesPointer = PropertyAverageValuesPointer
            NewNpropperTime = NpropperTime
            NewNpropperTimeStatic = NpropperTimeStatic
            NewNpropperTimeDynamic = NpropperTimeDynamic

            #          NewNormedInputStaticProps = NormedInputStaticProps
            NewNormedDynamicPropertyTimeSeries = NormedDynamicPropertyTimeSeries

        # Order of Predictions *****************************
        # Basic "futured" Predictions from property dynamic arrays
        # Additional predictions without futures and NOT in property arrays including Calculated time series
        # LengthFutures predictions for first NumpredFuturedperTime predictions
        # Special predictions (temporal, positional) added later
        NpredperTime = NumpredbasicperTime + NumpredFuturedperTime * LengthFutures
        Npredperseq = NpredperTime
        Predictionbasicname = [' '] * NumpredbasicperTime
        for ipred in range(0, NumpredbasicperTime):
            if PredSource[ipred] == 'Dynamic':
                Predictionbasicname[ipred] = InputPropertyNames[PredSourceNumber[ipred] + NpropperTimeStatic]

        TotalFutures = 0
        if NumpredFuturedperTime <= 0:
            GenerateFutures = False
        if GenerateFutures:
            TotalFutures = NumpredFuturedperTime * LengthFutures
        print(startbold + 'Predictions Total ' + str(Npredperseq) + ' Basic ' + str(
            NumpredbasicperTime) + ' Of which futured are '
              + str(NumpredFuturedperTime) + ' Giving number explicit futures ' + str(TotalFutures) + resetfonts)
        Predictionname = [' '] * Npredperseq
        Predictionnametype = [' '] * Npredperseq
        Predictionoldvalue = np.empty(Npredperseq, dtype=int)
        Predictionnewvalue = np.empty(Npredperseq, dtype=int)
        Predictionday = np.empty(Npredperseq, dtype=int)
        PredictionAverageValuesPointer = np.empty(Npredperseq, dtype=int)
        Predictionwgt = [1.0] * Npredperseq
        for ipred in range(0, NumpredbasicperTime):
            Predictionnametype[ipred] = PredSource[ipred]
            Predictionoldvalue[ipred] = PredSourceNumber[ipred]
            Predictionnewvalue[ipred] = ipred
            if PredSource[ipred] == 'Dynamic':
                PredictionAverageValuesPointer[ipred] = NpropperTimeStatic + Predictionoldvalue[ipred]
            else:
                PredictionAverageValuesPointer[ipred] = NpropperTime + PredSourceNumber[ipred]
            Predictionwgt[ipred] = 1.0
            Predictionday[ipred] = 1
            extrastring = ''
            Predictionname[ipred] = 'Next ' + Predictionbasicname[ipred]
            if FuturedPred[ipred] >= 0:
                extrastring = ' Explicit Futures Added '
            print(str(ipred) + ' Internal Property # ' + str(PredictionAverageValuesPointer[ipred]) + ' ' +
                  Predictionname[ipred]
                  + ' Weight ' + str(round(Predictionwgt[ipred], 3)) + ' Day ' + str(
                Predictionday[ipred]) + extrastring)

        for ifuture in range(0, LengthFutures):
            for ipred in range(0, NumpredbasicperTime):
                if FuturedPred[ipred] >= 0:
                    FuturedPosition = NumpredbasicperTime + NumpredFuturedperTime * ifuture + FuturedPred[ipred]
                    Predictionname[FuturedPosition] = Predictionbasicname[ipred] + Futures[ifuture].name
                    Predictionday[FuturedPosition] = Futures[ifuture].days[0]
                    Predictionwgt[FuturedPosition] = Futures[ifuture].classweight
                    Predictionnametype[FuturedPosition] = Predictionnametype[ipred]
                    Predictionoldvalue[FuturedPosition] = Predictionoldvalue[ipred]
                    Predictionnewvalue[FuturedPosition] = Predictionnewvalue[ipred]
                    PredictionAverageValuesPointer[FuturedPosition] = NpropperTimeStatic + \
                                                                      PredictionAverageValuesPointer[ipred]
                    print(str(iprop) + ' Internal Property # ' + str(
                        PredictionAverageValuesPointer[FuturedPosition]) + ' ' +
                          Predictionname[FuturedPosition] + ' Weight ' + str(round(Predictionwgt[FuturedPosition], 3))
                          + ' Day ' + str(Predictionday[FuturedPosition]) + ' This is Explicit Future')

        Predictionnamelookup = {}
        print(startbold + '\nBasic Predicted Quantities' + resetfonts)
        for i in range(0, Npredperseq):
            Predictionnamelookup[Predictionname[i]] = i

            iprop = Predictionnewvalue[i]
            line = Predictionbasicname[iprop]
            line += ' Weight ' + str(round(Predictionwgt[i], 4))
            if (iprop < NumpredFuturedperTime) or (iprop >= NumpredbasicperTime):
                line += ' Day= ' + str(Predictionday[i])
                line += ' Name ' + Predictionname[i]
            print(line)

            # Note that only Predictionwgt and Predictionname defined for later addons

        if PredictionsfromInputs:
            InputPredictionsbyTime = np.zeros([Num_Time, Nloc, Npredperseq], dtype=np.float32)
            for ipred in range(0, NumpredbasicperTime):
                if Predictionnametype[ipred] == 'Dynamic':
                    InputPredictionsbyTime[:, :, ipred] = NormedDynamicPropertyTimeSeries[:, :,
                                                          Predictionoldvalue[ipred]]

            # Add Futures based on Futured properties
            if LengthFutures > 0:
                NaNall = np.full([Nloc], NaN, dtype=np.float32)
                daystartveto = 0
                atendveto = 0
                allok = NumpredbasicperTime
                for ifuture in range(0, LengthFutures):
                    for itime in range(0, Num_Time):
                        ActualTime = itime + Futures[ifuture].days[0] - 1
                        if ActualTime >= Num_Time:
                            for ipred in range(0, NumpredbasicperTime):
                                Putithere = FuturedPred[ipred]
                                if Putithere >= 0:
                                    InputPredictionsbyTime[itime, :,
                                    NumpredbasicperTime + NumpredFuturedperTime * ifuture + Putithere] = NaNall
                            atendveto += 1

                        else:
                            for ipred in range(0, NumpredbasicperTime):
                                Putithere = FuturedPred[ipred]
                                if Putithere >= 0:
                                    if Predictionnametype[ipred] == 'Dynamic':
                                        InputPredictionsbyTime[itime, :,
                                        NumpredbasicperTime + NumpredFuturedperTime * ifuture + Putithere] \
                                            = NormedDynamicPropertyTimeSeries[ActualTime, :, Predictionoldvalue[ipred]]

                            allok += NumpredFuturedperTime
                print(startbold + 'Futures Added: Predictions set from inputs OK ' + str(allok) +
                      ' Veto at end ' + str(atendveto) + ' Veto at start ' + str(
                    daystartveto) + ' Times number of locations' + resetfonts)

        # Clean-up Input Source
        if len(InputSource) > 0:
            PropertyNameIndex = NewPropertyNameIndex
            NewPropertyNameIndex = None
            PropertyAverageValuesPointer = NewPropertyAverageValuesPointer
            NewPropertyAverageValuesPointer = None

            NormedInputStaticProps = NewNormedInputStaticProps
            NewNormedInputStaticProps = None
            NormedDynamicPropertyTimeSeries = NewNormedDynamicPropertyTimeSeries
            NewNormedDynamicPropertyTimeSeries = None

            NpropperTime = NewNpropperTime
            NpropperTimeStatic = NewNpropperTimeStatic
            NpropperTimeDynamic = NewNpropperTimeDynamic

        print('Static Properties')
        if NpropperTimeStatic > 0:
            checkNaN(NormedInputStaticProps)
        else:
            print(' None Defined')
        print('Dynamic Properties')

        checkNaN(NormedDynamicPropertyTimeSeries)

        Num_SeqExtraUsed = Tseq - 1
        Num_Seq = Num_Time - Tseq
        Num_SeqPred = Num_Seq
        TSeqPred = Tseq
        TFTExtraTimes = 0
        Num_TimeTFT = Num_Time
        if TFTConversion:
            TFTExtraTimes = 1 + LengthFutures
            SymbolicWindows = True
            Num_SeqExtraUsed = Tseq  # as last position needed in input
            Num_TimeTFT = Num_Time + TFTExtraTimes
            Num_SeqPred = Num_Seq
            TseqPred = Tseq

        # If SymbolicWindows, sequences are not made but we use same array with that dimension (RawInputSeqDimension) set to 1
        # reshape can get rid of this irrelevant dimension
        # Predictions and Input Properties are associated with sequence number which is first time value used in sequence
        # if SymbolicWindows false then sequences are labelled by sequence # and contain time values from sequence # to sequence# + Tseq-1
        # if SymbolicWindows True then sequences are labelled by time # and contain one value. They are displaced by Tseq
        # If TFT Inputs and Predictions do NOT differ by Tseq
        # Num_SeqExtra extra positions in RawInputSequencesTOT for Symbolic windows True as need to store full window
        # TFTExtraTimes are extra times
        RawInputSeqDimension = Tseq
        Num_SeqExtra = 0
        if SymbolicWindows:
            RawInputSeqDimension = 1
            Num_SeqExtra = Num_SeqExtraUsed

        if GenerateSequences:
            UseProperties = np.full(NpropperTime, True, dtype=np.bool)

            Npropperseq = 0
            IndexintoPropertyArrays = np.empty(NpropperTime, dtype=np.int)
            for iprop in range(0, NpropperTime):
                if UseProperties[iprop]:
                    IndexintoPropertyArrays[Npropperseq] = iprop
                    Npropperseq += 1
            RawInputSequences = np.zeros([Num_Seq + Num_SeqExtra, Nloc, RawInputSeqDimension, Npropperseq],
                                         dtype=np.float32)
            RawInputPredictions = np.zeros([Num_SeqPred, Nloc, Npredperseq], dtype=np.float32)

            locationarray = np.empty(Nloc, dtype=np.float32)
            for iseq in range(0, Num_Seq + Num_SeqExtra):
                for windowposition in range(0, RawInputSeqDimension):
                    itime = iseq + windowposition
                    for usedproperty in range(0, Npropperseq):
                        iprop = IndexintoPropertyArrays[usedproperty]
                        if iprop >= NpropperTimeStatic:
                            jprop = iprop - NpropperTimeStatic
                            locationarray = NormedDynamicPropertyTimeSeries[itime, :, jprop]
                        else:
                            locationarray = NormedInputStaticProps[:, iprop]
                        RawInputSequences[iseq, :, windowposition, usedproperty] = locationarray
                if iseq < Num_SeqPred:
                    RawInputPredictions[iseq, :, :] = InputPredictionsbyTime[iseq + TseqPred, :, :]
            print(startbold + 'Sequences set from Time values Num Seq ' + str(Num_SeqPred) + ' Time ' + str(
                Num_Time) + resetfonts)

        NormedInputTimeSeries = None
        NormedDynamicPropertyTimeSeries = None
        if GarbageCollect:
            gc.collect()

        GlobalTimeMask = np.empty([1, 1, 1, Tseq, Tseq], dtype=np.float32)

        print("Total number of Time Units " + str(NumberofTimeunits) + ' day')
        if NumberofTimeunits != (Num_Seq + Tseq):
            printexit("EXIT Wrong Number of Time Units " + str(Num_Seq + Tseq))

        Dateslist = []
        for i in range(0, NumberofTimeunits + TFTExtraTimes):
            Dateslist.append(InitialDate + datetime.timedelta(days=i * Dailyunit))

        LinearoverLocationEncoding = LinearLocationEncoding(Nloc)
        LinearovertimeEncoding = LinearTimeEncoding(Dateslist)

        if Dailyunit == 1:
            CosWeeklytimeEncoding, SinWeeklytimeEncoding = WeeklyTimeEncoding(Dateslist)
        CosAnnualtimeEncoding, SinAnnualtimeEncoding = AnnualTimeEncoding(Dateslist)

        # Encodings

        # linearlocationposition
        # Supported Time Dependent Probes that can be in properties and/or predictions
        # Special
        # Annual
        # Weekly
        #
        # Top Down
        # TD0 Constant at 0.5
        # TD1 Linear from 0 to 1
        # TD2 P2(x) where x goes from -1 to 1 as time goes from start to end
        #
        # Bottom Up
        # n-way Cos and sin theta where n = 4 7 8 16 24 32

        EncodingTypes = {'Spatial': 1, 'Weekly': 2, 'Annual': 3, 'TopDown': 4, 'BottomUp': 5}

        PropIndex = []
        PropNameMeanStd = []
        PropMeanStd = []
        PropArray = []
        PropPosition = []

        PredIndex = []
        PredNameMeanStd = []
        PredArray = []
        PredPosition = []

        Numberpropaddons = 0
        propposition = Npropperseq
        Numberpredaddons = 0
        predposition = Npredperseq

        numprop = len(PropTypes)
        if numprop != len(PropValues):
            printexit('Error in property addons ' + str(numprop) + ' ' + str(len(PropValues)))
        for newpropinlist in range(0, numprop):
            Typeindex = EncodingTypes[PropTypes[newpropinlist]]
            a, b, c, d = ReturnEncoding(Num_Time + TFTExtraTimes, Typeindex, PropValues[newpropinlist], LinearoverLocationEncoding=LinearoverLocationEncoding,
                                        CosWeeklytimeEncoding=CosWeeklytimeEncoding, SinWeeklytimeEncoding=SinWeeklytimeEncoding,
                                        CosAnnualtimeEncoding=CosAnnualtimeEncoding, SinAnnualtimeEncoding=SinAnnualtimeEncoding, LinearovertimeEncoding=LinearovertimeEncoding)
            if c[0] != 'Dummy':
                PropIndex.append(Typeindex)
                PropNameMeanStd.append(c)
                InputPropertyNames.append(c[0])
                PropArray.append(a)
                PropPosition.append(propposition)
                propposition += 1
                Numberpropaddons += 1
                line = ' '
                for ipr in range(0, 20):
                    line += str(round(a[ipr], 4)) + ' '
            #    print('c'+line)
            if d[0] != 'Dummy':
                PropIndex.append(Typeindex)
                PropNameMeanStd.append(d)
                InputPropertyNames.append(d[0])
                PropArray.append(b)
                PropPosition.append(propposition)
                propposition += 1
                Numberpropaddons += 1
                line = ' '
                for ipr in range(0, 20):
                    line += str(round(b[ipr], 4)) + ' '
        #    print('d'+line)

        numpred = len(PredTypes)
        if numpred != len(PredValues):
            printexit('Error in prediction addons ' + str(numpred) + ' ' + str(len(PredValues)))
        for newpredinlist in range(0, numpred):
            Typeindex = EncodingTypes[PredTypes[newpredinlist]]
            a, b, c, d = ReturnEncoding(Num_Time + TFTExtraTimes, Typeindex, PredValues[newpredinlist])
            if c[0] != 'Dummy':
                PredIndex.append(Typeindex)
                PredNameMeanStd.append(c)
                PredArray.append(a)
                Predictionname.append(c[0])
                Predictionnamelookup[c] = predposition
                PredPosition.append(predposition)
                predposition += 1
                Numberpredaddons += 1
                Predictionwgt.append(0.25)
            if d[0] != 'Dummy':
                PredIndex.append(Typeindex)
                PredNameMeanStd.append(d)
                PredArray.append(b)
                Predictionname.append(d[0])
                Predictionnamelookup[d[0]] = predposition
                PredPosition.append(predposition)
                predposition += 1
                Numberpredaddons += 1
                Predictionwgt.append(0.25)

        NpropperseqTOT = Npropperseq + Numberpropaddons

        # These include both Property and Prediction Variables
        NpropperTimeMAX = len(QuantityTakeroot)
        NewNpropperTimeMAX = NpropperTimeMAX + Numberpropaddons + Numberpredaddons
        NewQuantityStatistics = np.zeros([NewNpropperTimeMAX, 7], dtype=np.float32)
        NewQuantityTakeroot = np.full(NewNpropperTimeMAX, 1, dtype=np.int)  # All new ones aare 1 and are set here
        NewQuantityStatistics[0:NpropperTimeMAX, :] = QuantityStatistics[0:NpropperTimeMAX, :]
        NewQuantityTakeroot[0:NpropperTimeMAX] = QuantityTakeroot[0:NpropperTimeMAX]

        # Lookup for property names
        NewPropertyNameIndex = np.empty(NpropperseqTOT, dtype=np.int32)
        NumberofNames = len(InputPropertyNames) - Numberpropaddons
        NewPropertyNameIndex[0:Npropperseq] = PropertyNameIndex[0:Npropperseq]

        NewPropertyAverageValuesPointer = np.empty(NpropperseqTOT, dtype=np.int32)
        NewPropertyAverageValuesPointer[0:Npropperseq] = PropertyAverageValuesPointer[0:Npropperseq]

        for propaddons in range(0, Numberpropaddons):
            NewPropertyNameIndex[Npropperseq + propaddons] = NumberofNames + propaddons
            NewPropertyAverageValuesPointer[Npropperseq + propaddons] = NpropperTimeMAX + propaddons
            NewQuantityStatistics[NpropperTimeMAX + propaddons, :] = SetNewAverages(PropNameMeanStd[propaddons])

        # Set extra Predictions metadata for Sequences
        NpredperseqTOT = Npredperseq + Numberpredaddons

        NewPredictionAverageValuesPointer = np.empty(NpredperseqTOT, dtype=np.int32)
        NewPredictionAverageValuesPointer[0:Npredperseq] = PredictionAverageValuesPointer[0:Npredperseq]

        for predaddons in range(0, Numberpredaddons):
            NewPredictionAverageValuesPointer[
                Npredperseq + predaddons] = NpropperTimeMAX + +Numberpropaddons + predaddons
            NewQuantityStatistics[NpropperTimeMAX + Numberpropaddons + predaddons, :] = SetNewAverages(
                PredNameMeanStd[predaddons])

        RawInputSequencesTOT = np.empty(
            [Num_Seq + Num_SeqExtra + TFTExtraTimes, Nloc, RawInputSeqDimension, NpropperseqTOT], dtype=np.float32)
        flsize = np.float(Num_Seq + Num_SeqExtra) * np.float(Nloc) * np.float(RawInputSeqDimension) * np.float(
            NpropperseqTOT) * 4.0
        print('Total storage ' + str(round(flsize, 0)) + ' Bytes')

        for i in range(0, Num_Seq + Num_SeqExtra):
            for iprop in range(0, Npropperseq):
                RawInputSequencesTOT[i, :, :, iprop] = RawInputSequences[i, :, :, iprop]
        for i in range(Num_Seq + Num_SeqExtra, Num_Seq + Num_SeqExtra + TFTExtraTimes):
            for iprop in range(0, Npropperseq):
                RawInputSequencesTOT[i, :, :, iprop] = NaN

        for i in range(0, Num_Seq + Num_SeqExtra + TFTExtraTimes):
            for k in range(0, RawInputSeqDimension):
                for iprop in range(0, Numberpropaddons):
                    if PropIndex[iprop] == 1:
                        continue
                    RawInputSequencesTOT[i, :, k, PropPosition[iprop]] = PropArray[iprop][i + k]

        for iprop in range(0, Numberpropaddons):
            if PropIndex[iprop] == 1:
                for j in range(0, Nloc):
                    RawInputSequencesTOT[:, j, :, PropPosition[iprop]] = PropArray[iprop][j]

        # Set extra Predictions for Sequences
        RawInputPredictionsTOT = np.empty([Num_SeqPred + TFTExtraTimes, Nloc, NpredperseqTOT], dtype=np.float32)

        for i in range(0, Num_SeqPred):
            for ipred in range(0, Npredperseq):
                RawInputPredictionsTOT[i, :, ipred] = RawInputPredictions[i, :, ipred]
        for i in range(Num_SeqPred, Num_SeqPred + TFTExtraTimes):
            for ipred in range(0, Npredperseq):
                RawInputPredictionsTOT[i, :, ipred] = NaN

        for i in range(0, Num_SeqPred + TFTExtraTimes):
            for ipred in range(0, Numberpredaddons):
                if PredIndex[ipred] == 1:
                    continue
                actualarray = PredArray[ipred]
                RawInputPredictionsTOT[i, :, PredPosition[ipred]] = actualarray[i + TseqPred]

        for ipred in range(0, Numberpredaddons):
            if PredIndex[ipred] == 1:
                for j in range(0, Nloc):
                    RawInputPredictionsTOT[:, j, PredPosition[ipred]] = PredArray[ipred][j]

        PropertyNameIndex = None
        PropertyNameIndex = NewPropertyNameIndex
        QuantityStatistics = None
        QuantityStatistics = NewQuantityStatistics
        QuantityTakeroot = None
        QuantityTakeroot = NewQuantityTakeroot
        PropertyAverageValuesPointer = None
        PropertyAverageValuesPointer = NewPropertyAverageValuesPointer
        PredictionAverageValuesPointer = None
        PredictionAverageValuesPointer = NewPredictionAverageValuesPointer

        print('Time and Space encoding added to input and predictions')

        if SymbolicWindows:
            SymbolicInputSequencesTOT = np.empty([Num_Seq, Nloc], dtype=np.int32)  # This is sequences
            for iseq in range(0, Num_Seq):
                for iloc in range(0, Nloc):
                    SymbolicInputSequencesTOT[iseq, iloc] = np.left_shift(iseq, 16) + iloc
            ReshapedSequencesTOT = np.transpose(RawInputSequencesTOT, (1, 0, 3, 2))
            ReshapedSequencesTOT = np.reshape(ReshapedSequencesTOT,
                                              (Nloc, Num_Seq + Num_SeqExtra + TFTExtraTimes, NpropperseqTOT))

        # To calculate masks (identical to Symbolic windows)
        SpacetimeforMask = np.empty([Num_Seq, Nloc], dtype=np.int32)
        for iseq in range(0, Num_Seq):
            for iloc in range(0, Nloc):
                SpacetimeforMask[iseq, iloc] = np.left_shift(iseq, 16) + iloc

        print(PropertyNameIndex)
        print(InputPropertyNames)
        for iprop in range(0, NpropperseqTOT):
            print('Property ' + str(iprop) + ' ' + InputPropertyNames[PropertyNameIndex[iprop]])
        for ipred in range(0, NpredperseqTOT):
            print('Prediction ' + str(ipred) + ' ' + Predictionname[ipred] + ' ' + str(round(Predictionwgt[ipred], 3)))

        RawInputPredictions = None
        RawInputSequences = None
        if SymbolicWindows:
            RawInputSequencesTOT = None
        if GarbageCollect:
            gc.collect()

        # Location based validation from Colab notebook - incomplete
        # LocationBasedValidation = True
        # LocationValidationFraction = 0.0
        # RestartLocationBasedValidation = False
        # RestartRunName = RunName
        # RestartRunName = 'EARTHQN-Transformer3'
        # FullSetValidation = False
        #
        # global SeparateValandTrainingPlots
        # SeparateValandTrainingPlots = True
        # if not LocationBasedValidation:
        #     SeparateValandTrainingPlots = False
        #     LocationValidationFraction = 0.0
        # NlocValplusTraining = Nloc
        # ListofTrainingLocs = np.arange(Nloc, dtype=np.int32)
        # ListofValidationLocs = np.full(Nloc, -1, dtype=np.int32)
        # MappingtoTraining = np.arange(Nloc, dtype=np.int32)
        # MappingtoValidation = np.full(Nloc, -1, dtype=np.int32)
        # TrainingNloc = Nloc
        # ValidationNloc = 0
        # if LocationBasedValidation:
        #     if RestartLocationBasedValidation:
        #         InputFileName = APPLDIR + '/Validation' + RestartRunName
        #         with open(InputFileName, 'r', newline='') as inputfile:
        #             Myreader = reader(inputfile, delimiter=',')
        #             header = next(Myreader)
        #             LocationValidationFraction = np.float32(header[0])
        #             TrainingNloc = np.int32(header[1])
        #             ValidationNloc = np.int32(header[2])
        #
        #             ListofTrainingLocs = np.empty(TrainingNloc, dtype=np.int32)
        #             ListofValidationLocs = np.empty(ValidationNloc, dtype=np.int32)
        #             nextrow = next(Myreader)
        #             for iloc in range(0, TrainingNloc):
        #                 ListofTrainingLocs[iloc] = np.int32(nextrow[iloc])
        #             nextrow = next(Myreader)
        #             for iloc in range(0, ValidationNloc):
        #                 ListofValidationLocs[iloc] = np.int32(nextrow[iloc])
        #
        #         LocationTrainingfraction = 1.0 - LocationValidationFraction
        #         if TrainingNloc + ValidationNloc != Nloc:
        #             printexit('EXIT: Inconsistent location counts for Location Validation ' + str(Nloc)
        #                       + ' ' + str(TrainingNloc) + ' ' + str(ValidationNloc))
        #         print(' Validation restarted Fraction ' + str(
        #             round(LocationValidationFraction, 4)) + ' ' + RestartRunName)
        #
        #     else:
        #         LocationTrainingfraction = 1.0 - LocationValidationFraction
        #         TrainingNloc = math.ceil(LocationTrainingfraction * Nloc)
        #         ValidationNloc = Nloc - TrainingNloc
        #         np.random.shuffle(ListofTrainingLocs)
        #         ListofValidationLocs = ListofTrainingLocs[TrainingNloc:Nloc]
        #         ListofTrainingLocs = ListofTrainingLocs[0:TrainingNloc]
        #
        #     for iloc in range(0, TrainingNloc):
        #         jloc = ListofTrainingLocs[iloc]
        #         MappingtoTraining[jloc] = iloc
        #         MappingtoValidation[jloc] = -1
        #     for iloc in range(0, ValidationNloc):
        #         jloc = ListofValidationLocs[iloc]
        #         MappingtoValidation[jloc] = iloc
        #         MappingtoTraining[jloc] = -1
        #     if ValidationNloc <= 0:
        #         SeparateValandTrainingPlots = False
        #
        #
        #
        #     print('Training Locations ' + str(TrainingNloc) + ' Validation Locations ' + str(ValidationNloc))
        #     if ValidationNloc <= 0:
        #         LocationBasedValidation = False

        Totalsize = (Num_Time + TFTExtraTimes) * Nloc
        RawLabel = np.arange(0, Totalsize, dtype=np.float32)
        LocationLabel = []
        FFFFWNPFUniqueLabel = []
        RawTime = np.empty([Nloc, Num_Time + TFTExtraTimes], dtype=np.float32)
        #  print('Times ' + str(Num_Time) + ' ' + str(TFTExtraTimes))
        ierror = 0
        for ilocation in range(0, Nloc):
            #   locname = Locationstate[LookupLocations[ilocation]] + ' ' + Locationname[LookupLocations[ilocation]]
            locname = Locationname[LookupLocations[ilocation]] + ' ' + Locationstate[LookupLocations[ilocation]]
            if locname == "":
                printexit('Illegal null location name ' + str(ilocation))
            for idupe in range(0, len(FFFFWNPFUniqueLabel)):
                if locname == FFFFWNPFUniqueLabel[idupe]:
                    print(' Duplicate location name ' + str(ilocation) + ' ' + str(idupe) + ' ' + locname)
                    ierror += 1
            FFFFWNPFUniqueLabel.append(locname)
            #    print(str(ilocation) + ' ' +locname)
            for jtime in range(0, Num_Time + TFTExtraTimes):
                RawTime[ilocation, jtime] = np.float32(jtime)
                LocationLabel.append(locname)
        if ierror > 0:
            printexit(" Duplicate Names " + str(ierror))
        RawTime = np.reshape(RawTime, -1)
        TFTdf1 = pd.DataFrame(RawLabel, columns=['RawLabel'])
        TFTdf2 = pd.DataFrame(LocationLabel, columns=['Location'])
        TFTdf3 = pd.DataFrame(RawTime, columns=['Time from Start'])
        TFTdfTotal = pd.concat([TFTdf1, TFTdf2, TFTdf3], axis=1)

        # left off here
        #PropPick = [1 for i in range(len(self.config['inputs_use']) + 2)]
        PropPick = [1 for i in range(NpropperseqTOT)]
        print('Properties in use ' + str(len(PropPick)))
        print('Properties processed ' + str(NpropperseqTOT))
        ColumnsProp = []
        for iprop in range(0, NpropperseqTOT):
            line = str(iprop) + ' ' + InputPropertyNames[PropertyNameIndex[iprop]]
            jprop = PropertyAverageValuesPointer[iprop]
            if QuantityTakeroot[jprop] > 1:
                line += ' Root ' + str(QuantityTakeroot[jprop])
            ColumnsProp.append(line)

        TFTInputSequences = np.reshape(ReshapedSequencesTOT, (-1, NpropperseqTOT))
        TFTNumberTargets = 0
        for iprop in range(0, NpropperseqTOT):
            if PropPick[iprop] >= 0:
                if PropPick[iprop] == 0:
                    TFTNumberTargets += 1
                nextcol = TFTInputSequences[:, iprop]
                dftemp = pd.DataFrame(nextcol, columns=[ColumnsProp[iprop]])
                TFTdfTotal = pd.concat([TFTdfTotal, dftemp], axis=1)

        FFFFWNPFNumberTargets = TFTNumberTargets


        return TFTdfTotal
