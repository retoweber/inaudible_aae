# -*- coding: utf-8 -*-
import numpy as np

def quiet(f = 1000.0, method='default'):
  # Audio Watermark Chapter 2.2.3: Eq 2.2
  if(f < 20) : # there were wierd artifact in the low frequency domain.
    f = 20
  f = f/1000.0
  return 3.64 * pow(f, -0.8) - 6.5 * np.exp(-0.6 * pow(f - 3.3, 2)) + 0.001 * pow(f, 4)
  
def hzToBark(f = 1000, method='default'):
    # Fastl & Zwicker: Psychoacoustics: Facts and Models.
    if(method=='default'):
        return 13.0*np.arctan(0.00076*f) + 3.5*np.arctan((f/7500.0)*(f/7500.0))
    elif(method=='Traunmueller'):
        return ((26.81*f)/(1960.0+f))-0.53
    elif(mothod=='Wang'):
        return 6*np.arcsinh(f/600.0)
    else:
      return float('nan')
      
def barkToHz(bark):
    # binary search using hzToBark as I was not able to invert the function
    p0 = 0
    p1 = 8000
    b0 = hzToBark(p0)
    b1 = hzToBark(p1)
    pivot = (p0+p1)/2
    b_pivot = hzToBark(pivot)
    while(p1-p0>0.1):
        if(bark < b_pivot):
            p1 = pivot
            pivot = (p0+p1)/2
            b_pivot = hzToBark(pivot)
        elif(bark > b_pivot):
            p0 = pivot
            pivot = (p0+p1)/2
            b_pivot = hzToBark(pivot)
        else: 
            break
    return pivot

def frequencyMasking(Sft = 60, dHz = 0, dBark = 0, method='tristan simplified'):
  # df is the smalles unit of frequency (Hz) (frequency resolution)
  # dBark is the smalles unit of frequency (Bark) (frequency resolution)
  if(method=='tristan simple'):
    if(dBark<0):
      return dBark * 25
    else:
      return -dBark * 10
  elif(method=='tristan precise'):
    BW = 100 if(dHz<500) else 0.2*dHz
    z = dBark
    i = min(5.0 * Sft * BW, 2.0)
    return (15.81 - i) + 7.5 * (z+0.474) - (17.5 -i)*np.sqrt(1 + (z + 0.474)**2)
  elif(method=='tristan simplified'):
    ref = 15.81 + 7.5*(0.474) - 17.5*np.sqrt(1 + (0.474)**2)
    #return 13.81 + 7.5*(dBark+0.474) - 15.5*np.sqrt(1 + (dBark + 0.474)**2)
    return 15.81 + 7.5*(dBark+0.474) - 17.5*np.sqrt(1 + (dBark + 0.474)**2) - ref


def forwardMasking(Sft = 60, dt = 0, method='tristan'):
  #minTh = -20.
  #factor = (Sft - minTh)/Sft
  # dt in ms
  
  if(method=='dai soon'):
    # simplified model used in "A temporal frequency warped (TFW) 2D psychoacoustic filter for robust speech recognition system"
    # inspired by : Jesteadt, W., Bacon, S., Lehman, J., 1982. 
    # Forward masking as a function of frequency, masker level, and signal delay. 
    # J. Acoust. Soc. Amer. 71, 950–962
    a,b = [0.351, 2.252]
    if(dt>0):
      return np.maximum(a * (b - np.log10(dt)) * Sft, 0)
    else:
      return 0
  if(method=='tristan'):
    if(dt>=0 and dt <= 200.0):
      return Sft*(np.cos(np.pi*dt/2/200.0)**2)
    else:
      return 0

def backwardMasking(Sft = 60, dt = 0, method='tristan'):
  if(method=='tristan'):
    if(dt<=0 and dt >= -20.0):
      dt=-dt
      return Sft*(np.cos(np.pi*dt/2/20.0)**2)
    else:
      dt=-dt
      return 0


def temporalMasking(Sft = 60, dt = 0, methods = ('tristan', 'dai soon')):
  if(dt<0):
    return backwardMasking(Sft, dt, methods[0])
  elif(dt==0):
    return 1
  else:
    return forwardMasking(Sft, dt, methods[1])

def totalMask(Sft = 60, dHz = 0, dBark = 0, method='default'):
  if(method=='default'):
    return Sft + temporalMasking(Sft, dt) + frequencyMasking(Sft, dBark=dBark, method='tristan simplified')
  else:
    return Sft + temporalMasking(Sft, dt) + frequencyMasking(Sft, dBark=dBark, method='tristan simple')
