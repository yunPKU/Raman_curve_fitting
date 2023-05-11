import os
import math
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize, signal,integrate
from lmfit import models


def gauss_crv(x, A, μ, σ):
    return A / (σ * math.sqrt(2 * math.pi)) * np.exp(-(x-μ)**2 / (2*σ**2))


def generate_model(spec):
    composite_model = None
    params = None
    x = spec['x']
    y = spec['y']
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    y_max = np.max(y)
    for i, basis_func in enumerate(spec['model']):
        prefix = f'm{i}_'
        model = getattr(models, basis_func['type'])(prefix=prefix)
        if basis_func['type'] in ['GaussianModel', 'LorentzianModel', 'VoigtModel']: # for now VoigtModel has gamma constrained to sigma
            model.set_param_hint('sigma', min=1e-6, max=x_range)
            model.set_param_hint('center', min=x_min, max=x_max)
            model.set_param_hint('height', min=1e-6, max=1.1*y_max)
            model.set_param_hint('amplitude', min=1e-6)
            # default guess is horrible!! do not use guess()
            default_params = {
                prefix+'center': x_min + x_range * random.random(),
                prefix+'height': y_max * random.random(),
                prefix+'sigma': x_range * random.random()
            }
        else:
            raise NotImplemented(f'model {basis_func["type"]} not implemented yet')
        if 'help' in basis_func:  # allow override of settings in parameter
            for param, options in basis_func['help'].items():
                model.set_param_hint(param, **options)
        model_params = model.make_params(**default_params, **basis_func.get('params', {}))
        if params is None:
            params = model_params
        else:
            params.update(model_params)
        if composite_model is None:
            composite_model = model
        else:
            composite_model = composite_model + model
    return composite_model, params

def update_spec_from_peaks(spec, model_indicies, peak_widths=(10, 25), **kwargs):
    x = spec['x']
    y = spec['y']
    x_range = np.max(x) - np.min(x)
    peak_indicies = signal.find_peaks_cwt(y, peak_widths)
    np.random.shuffle(peak_indicies)
    for peak_indicie, model_indicie in zip(peak_indicies.tolist(), model_indicies):
        model = spec['model'][model_indicie]
        if model['type'] in ['GaussianModel', 'LorentzianModel', 'VoigtModel']:
            params = {
                'height': y[peak_indicie],
                'sigma': x_range / len(x) * np.min(peak_widths),
                'center': x[peak_indicie]
            }
            if 'params' in model:
                model.update(params)
            else:
                model['params'] = params
        else:
            raise NotImplemented(f'model {basis_func["type"]} not implemented yet')
    return peak_indicies
    
def update_spec_from_peaks(spec, model_indicies, peak_widths=(10, 25), **kwargs):
    x = spec['x']
    y = spec['y']
    x_range = np.max(x) - np.min(x)
    peak_indicies = signal.find_peaks_cwt(y, peak_widths)
    np.random.shuffle(peak_indicies)
    for peak_indicie, model_indicie in zip(peak_indicies.tolist(), model_indicies):
        model = spec['model'][model_indicie]
        if model['type'] in ['GaussianModel', 'LorentzianModel', 'VoigtModel']:
            params = {
                'height': y[peak_indicie],
                'sigma': x_range / len(x) * np.min(peak_widths),
                'center': x[peak_indicie]
            }
            if 'params' in model:
                model.update(params)
            else:
                model['params'] = params
        else:
            raise NotImplemented(f'model {basis_func["type"]} not implemented yet')
    return peak_indicies
def print_best_values(spec, output):
    model_params = {
        'GaussianModel':   ['amplitude', 'sigma'],
        'LorentzianModel': ['amplitude', 'sigma'],
        'VoigtModel':      ['amplitude', 'sigma', 'gamma']
    }
    best_values = output.best_values
    print('center    model   amplitude     sigma      gamma')
    for i, model in enumerate(spec['model']):
        prefix = f'm{i}_'
        values = ', '.join(f'{best_values[prefix+param]:8.3f}' for param in model_params[model["type"]])
        print(f'[{best_values[prefix+"center"]:3.3f}] {model["type"]:16}: {values}')
    print('RedChi = '+str(output.redchi))


def spec_fitting(fileName,st_cm,ed_cm,init_center,lowBnd,upBnd): 
    # return the fitting results including the center, amp, sigma, area for all the peaks and the redChi of fitting
    mltK = 50 # repeat the fitting for mltK times
    mulOutput = []
    mulredChi = np.zeros(mltK)
    prtRange = 40
    for i in range(mltK):
        raw_spec = np.loadtxt(fileName);
        x = raw_spec[(raw_spec[:,0]>st_cm) & (raw_spec[:,0]<ed_cm), 0]
        y = raw_spec[(raw_spec[:,0]>st_cm) & (raw_spec[:,0]<ed_cm), 1]
        
        # constructing the components
        spec = {
            'x': x,
            'y': y,
            'model': [
                {
                    'type': 'GaussianModel',
                    'params': {'center': init_center[0]+(np.random.rand()-0.5)*prtRange},
                    'help': {'center': {'min': lowBnd[0], 'max': upBnd[0]}}
                },
                {
                    'type': 'GaussianModel',
                    'params': {'center': init_center[1]+(np.random.rand()-0.5)*prtRange},
                    'help': {'center': {'min': lowBnd[1], 'max': upBnd[1]}}
                },
                {
                    'type': 'GaussianModel',
                    'params': {'center': init_center[2]+(np.random.rand()-0.5)*prtRange},
                    'help': {'center': {'min': lowBnd[2], 'max': upBnd[2]}}
                }    
            ]
        }
        # fitting for mltK times
        model, params = generate_model(spec)
        tmpoutput = model.fit(spec['y'], params, x=spec['x'])
        mulOutput += [tmpoutput]
        mulredChi[i] = tmpoutput.redchi;
    # find the fitting with the smallest redChi
    output = mulOutput[np.argmin(mulredChi)]
    fitRe  = pd.DataFrame([output.best_values])
    fitRe['redChi']   = output.redchi
    
    # calculate the area for each peak
    alArea = 0
    for i in range(len(init_center)):
        prefix = 'm'+str(i)+'_'
        Q_crv = lambda x:gauss_crv(x, output.best_values[prefix+'amplitude'], 
                        output.best_values[prefix+'center'], output.best_values[prefix+'sigma'])
        fitRe[prefix+'area'] = integrate.quad(Q_crv,st_cm,ed_cm)[0]
        alArea += fitRe[prefix+'area']
    
    for i in range(len(init_center)):
        prefix = 'm'+str(i)+'_'
        fitRe[prefix+'area'] = fitRe[prefix+'area']/alArea;
    fitRe = peak_sorting(fitRe,pkNum=3) # added on 2023-03 for sorting the peaks
    return fitRe

def nadir_src(spec,ini_point,src_range):
    inteSpec = spec[(spec[:,0]>=ini_point-src_range) & (spec[:,0]<=ini_point+src_range), :]
    nadir_ID = np.argmin(inteSpec[:,1]);
    return inteSpec[nadir_ID,0]

def positive_area_cal(spec,st_cm,ed_cm,src_range):
    # find the nadir point within st_cn+/-src_range and ed_cm+/-src_range: denoted as st_cal, ed_cal
    st_cal = nadir_src(spec,st_cm,src_range)
    ed_cal = nadir_src(spec,ed_cm,src_range)


    # using st_cal, ed_cal as the range of integration
    inteSpec = spec[(spec[:,0]>st_cal) & (spec[:,0]<ed_cal), :]
    m,n = np.shape(inteSpec)
    area = 0;
    for i in range(m-1):
        tmp_cm, tmp_hgt = inteSpec[i,:];
        if tmp_hgt >= 0:
            area += (inteSpec[i+1,0] - inteSpec[i,0])*(inteSpec[i+1,1] + inteSpec[i,1])/2;
        elif tmp_cm - st_cal < ed_cal - tmp_cm: # the negative point is closer to the start point st_cal
            area = 0; # re-initiate the integration
        else: 
            break;  # stop the integration
    return area;

def peak_sorting(fitRe,pkNum):
    centerClm = []
    areaClm   = []
    for i in range(pkNum):
        prefix = 'm'+str(i)+'_';
        centerClm += [prefix+'center'];
        areaClm   += [prefix+'area'];
    tmpCenter = fitRe[centerClm].values
    tmpArea   = fitRe[areaClm].values
    tmpInfor  = np.concatenate((tmpCenter,tmpArea));
    tmpInfor  = tmpInfor[:, tmpInfor[0, :].argsort()] # sorting the peaks
    tmpInfor[1,:] = tmpInfor[1,:]/np.sum(tmpInfor[1,:])
    return tmpInfor;

def Ip_cal(fileName,bnding_st,bnding_ed,strching_st,strching_ed,src_range):
    raw_spec = np.loadtxt(fileName)
    bnding_st = 250;
    bnding_ed = 650;

    strching_st = 820;
    strching_ed = 1230;
    src_range = 50;
    area_500 = positive_area_cal(raw_spec,bnding_st,bnding_ed,src_range)
    area_1000 = positive_area_cal(raw_spec,strching_st,strching_ed,src_range)
    
    return area_500/area_1000

# 3 peaks
st_cm = 830 
ed_cm = 1250 # st_cm and ed_cm define the  range of peak fitting
init_center = [950,1050,1150] # initial peaks
lowBnd = [st_cm,1000,1100]
upBnd  = [1000,1100,ed_cm]

# file information
# filePath = '/Users/macbjmu/Documents/research/lan_file/2021-07/2022-03-paper/imitationStudy/imiData-spec/txt/dianluyanghua/Sub300/' 
fileName = 'demo_curve'
fileType = '.txt'

# # peak fitting
resu = spec_fitting(filePath+fileName+fileType,st_cm,ed_cm,init_center,lowBnd,upBnd);
# the output is the positions(vQi) of three spectral components and their corresponding area ratios (Ai), that is
# [vQ1,vQ2,vQ34;
#  A1, A2, A34]

# # calculating the Ip value
Ip_val = Ip_cal(filePath+fileName+fileType,bnding_st=250,bnding_ed=650,strching_st=820,strching_ed=1250,src_range=50)
