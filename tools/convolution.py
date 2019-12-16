# -*- coding: utf-8 -*-
import numpy as np
from scipy import signal
from tools import audio
from tools import helper

freq_start = 0
freq_end = 8000

    
def timeMask(res = 1.0):
  sft=1.  # strength of signal from PSD
  # used 1 to create a mask. Mask contains factor of how much masking is present.
  time = np.concatenate([np.arange(0-res, -200, -res)[::-1], np.arange(0,200,res)])
  return np.array([audio.temporalMasking(sft, dt) for dt in time])
  
def freqMask(res = 1.0):
  sft=60. # strength of signal from PSD
  # in the used frequeny masking method sft is not used and just comutes the mask.
  barks = np.concatenate([np.arange(0-res, -11, -res)[::-1], np.arange(0,11,res)])
  res = np.array([audio.frequencyMasking(sft, dBark=dBark) for dBark in barks])
  res = res - res.max()
  return res

def convolve(sig, resFreq = 1.0, resTime = 1.0, quiet = None):
  fMask = freqMask(resFreq)
  tMask = timeMask(resTime)
  sigFreq = freqConv(sig, resFreq)
  sigTotal = helper.mul_conv_1d(sig, tMask, zeros=quiet)
  return np.maximum(sigFreq, sigTotal)
  
def convolveTime(sig, resFreq, resTime, quiet = None):
  tMask = timeMask(resTime)
  sig = helper.mul_conv_1d(sig, tMask, zeros=quiet)
  return sig
  
def convolveFreq(sig, resFreq, resTime, quiet = None):
  fMask = freqMask(resFreq)
  sig = freqConv(sig, resFreq)
  return sig
  
def convolveId(sig, resFreq, resTime, quiet = None):
    return sig
  
def convolveNoMask(sig, resFreq, resTime, quiet = None):
    return np.ones(sig.shape)*-float('inf')
  
def totalMask(sig, resFreq = 1.0, resTime = 1.0, freq_start = 0, freq_end = 8000):
    shape = sig.shape
    freqs = np.array([np.arange(freq_start,freq_end+resFreq/2,resFreq) for _ in range(shape[1])]).T
    q = np.array([audio.quiet(f) if f > 0.01 else 0 for f in freqs.reshape(-1)]).reshape(shape)
    th = convolve(sig, resFreq, resTime, q)
    #th = convolveTime(sig, resFreq, resTime, q)
    #th = convolveFreq(sig, resFreq, resTime, q)
    #th = convolveId(sig, resFreq, resTime, q)
    return np.maximum(th, q)

def allMasks(sig, resFreq = 1.0, resTime = 1.0, freq_start = 0, freq_end = 8000):
    shape = sig.shape
    freqs = np.array([np.arange(freq_start,freq_end+resFreq/2,resFreq) for _ in range(shape[1])]).T
    q = np.array([audio.quiet(f) if f > 0.01 else 0 for f in freqs.reshape(-1)]).reshape(shape)
    th1 = convolve(sig, resFreq, resTime, q)
    th2 = convolveTime(sig, resFreq, resTime, q)
    temp = th2-sig
    th3 = convolveFreq(sig, resFreq, resTime, q)
    th4 = convolveId(sig, resFreq, resTime, q)
    return [np.maximum(th1, q), np.maximum(th2, q), np.maximum(th3, q), np.maximum(th4, q)]

def freqConv(psd, resFreq):
    freqs = np.arange(psd[:,0].shape[0])*resFreq
    dBark = np.maximum(0.02,audio.hzToBark(freqs[-1])-audio.hzToBark(freqs[-2]))
    barks = np.arange(0,audio.hzToBark(8000)+.5*dBark,dBark)
    fMask = freqMask(dBark)
    
    psdBark = np.zeros((barks.shape[0], psd.shape[1]))
    temp = np.zeros(barks.shape[0])
    barki = 0
    hzi = 1
    
    
    for barki in range(barks.shape[0]):
        bark = barks[barki]
        freqClose = audio.barkToHz(bark)
        freqPos = np.argmin(abs(freqs - freqClose))
        freqi = np.argmin(np.array(list(map(lambda x: np.inf if x < 0 else x, freqs - freqClose))))
        freqj = np.argmin(np.array(list(map(lambda x: np.inf if x > 0 else -x, freqs - freqClose))))
        if(freqi == freqj):
            psdBark[barki] = psd[freqi]
        else:
            ratio = (freqClose-freqs[freqj])/(freqs[freqi]-freqs[freqj])
            psdBark[barki] = psd[freqj] + ratio*(psd[freqi] - psd[freqj])
    psdBark = helper.add_conv_1d(psdBark.T, fMask).T
    
    retPsd = np.zeros(psd.shape)
    for hzi in range(freqs.shape[0]):
        barki = np.argmin(abs(barks - audio.hzToBark(freqs[hzi])))
        retPsd[hzi] = psdBark[barki]
    return np.maximum(retPsd, psd)






