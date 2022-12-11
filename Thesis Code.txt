!pip install praat-parselmouth==0.4.0
!pip install dtw-python

from google.colab import drive
drive.mount('/content/drive')

import IPython.display as ipd
import os
import re
import parselmouth as pm
from parselmouth.praat import call
from dtw import *
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import seaborn as sns
import matplotlib.ticker as ticker

def wavplay(wavfile):
  ipd.display(ipd.Audio(wavfile))
  return

def datplay(x, fs):
  ipd.display(ipd.Audio(x,rate=fs, autoplay=False))
  return

class Label:
  def __init__(self, filename):
    self.filename = filename
    self.start = []
    self.end = []
    self.label = []
    self.length = 0
    root, ext = os.path.splitext(filename)
    if ext == '.lab':
      for line in open(self.filename, "r"):
        items = line.split()
        self.start.append(float(items[0]))
        self.end.append(float(items[1]))
        self.label.append(items[2])
        self.length = len(self.start)
    elif ext == '.LB':
      for line in open(self.filename, "r"):
          if re.match("^#", line):
            break
          items = line.split()
          self.start.append(float(items[0])/1000.0)
          self.label.append(items[1])
          self.end.append(float(items[2])/1000.0)
          self.length = len(self.start)

  def name(self):
    print("filename = " + self.filename)

def getNearestIndex(list, value):
  idx = np.abs(np.asarray(list) - value).argmin()
  return idx

def draw_pitch2(ax, pitch, lab):
  pitch_values = pitch.selected_array['frequency']
  pitch_values[pitch_values==0] = np.nan
  ax.plot(pitch.xs(), pitch_values, 'o', markersize=4, color='w')
  ax.plot(pitch.xs(), pitch_values, 'o', markersize=2)
  ax.set_yscale('log')
  ax.set_ylabel("F0 [Hz]")
  formatter = ticker.FormatStrFormatter('%.0f')
  ax.yaxis.set_major_formatter(formatter)
  ax.yaxis.set_minor_formatter(formatter)
  ax.grid(True, which='both', axis='y')
  ax.tick_params(which ="both", axis="both", labelsize=5, pad=0)
  '''
  # The average value for each interval is indicated by a horizontal line.
  for j in range(lab.length):
    m = call(pitch, "Get mean", lab.start[j], lab.end[j], "Hertz (logarithmic)")
    # print(lab.label[j], lab.start[j], lab.end[j], m, sep=' ') 
    if not math.isnan(m) and lab.label[j] != 'silB' and lab.label[j] != 'sp' and lab.label[j] != 'silE':
      ax.plot([lab.start[j], lab.end[j]], [m, m], color="r", linewidth=3)
  '''

def ExtractF0(s, floor, ceiling):
  floor = 75.0 if floor < 0.0 else floor
  ceiling = 600.0 if ceiling < 0.0 else ceiling 
  return s.to_pitch_ac(time_step=0.001,pitch_floor=floor, pitch_ceiling=ceiling,voicing_threshold=0.63)
  #return s.to_pitch(time_step=0.001,pitch_floor=floor, pitch_ceiling=ceiling)

def GetSlope(pitch, start, end):
  t = np.arange(start, end, 0.001)
  f = call(pitch, "List values at times", t, "Hertz", "linear")
  ff = ~np.isnan(f)
  t1 = t[ff]
  f1 = np.log2(f[ff])
  #print(t1)
  #print(f1)
  slope, intercept = np.polyfit(t1,f1,1)
  #print('slope=', slope, " intercept=", intercept)
  return slope

def DrawWaveformF0s3(wavname, floor1, ceiling1, floor2, ceiling2, duration):
  labname1 = re.sub(r'_(.*)\.wav', r'_PCL.lab', wavname)
  labname2 = re.sub(r'_(.*)\.wav', r'_\1_PCL.lab', wavname)
  figname = re.sub(r'_(.*)\.wav', r'_\1.png', wavname)
  snd0 = pm.Sound(wavname)
  snd = [ snd0.extract_left_channel(), snd0.extract_right_channel() ]
  lab = [ Label(labname1), Label(labname2) ]
  pitch = [ ExtractF0(snd[0], floor1, ceiling1), ExtractF0(snd[1], floor2, ceiling2)]

  fig = plt.figure(figsize=(10,10))
  ax_wav = []
  ax_f0 = []
  ax_wav.append( plt.subplot2grid((10, 1), (0, 0)) )
  ax_f0.append( plt.subplot2grid((10, 1), (1, 0), rowspan=3) )
  ax_wav.append( plt.subplot2grid((10, 1), (4, 0)) )
  ax_f0.append( plt.subplot2grid((10, 1), (5, 0), rowspan=3) )
  ax_f0r = plt.subplot2grid((10,1), (8, 0), rowspan=2) 

  ax_wav[0].get_shared_x_axes().join(*ax_wav, *ax_f0, ax_f0r)
  ax_wav[1].get_shared_x_axes().join(*ax_wav, *ax_f0, ax_f0r)
  ax_f0[0].get_shared_x_axes().join(*ax_wav, *ax_f0, ax_f0r)
  ax_f0[1].get_shared_x_axes().join(*ax_wav, *ax_f0, ax_f0r)
  ax_f0r.get_shared_x_axes().join(*ax_wav, *ax_f0, ax_f0r)
  ax_wav[0].get_shared_y_axes().join(*ax_wav)
  ax_wav[1].get_shared_y_axes().join(*ax_wav)
  ax_f0[0].get_shared_y_axes().join(*ax_f0)
  ax_f0[1].get_shared_y_axes().join(*ax_f0)

  ax_f0r.set_yscale('log', basey=2)
  ax_f0r.set_ylim(0.4,3.0)

  for i in [0, 1]:
    ax_wav[i].plot(snd[i].xs(), snd[i].values.T,color="gray", linewidth=1)
    ax_wav[i].set_xlabel("time [s]")
    ax_wav[i].set_ylabel("amplitude")
    for j in range(lab[i].length):
      lab[i].start[j] = lab[i].end[j] - duration
      ax_wav[i].axvline(lab[i].start[j], color="b", linewidth=1)
      ax_wav[i].axvline(lab[i].end[j], color="b", linewidth=1)
      ax_wav[i].text((lab[i].start[j]+lab[i].end[j])*0.5, 0.5, lab[i].label[j], horizontalalignment="center", verticalalignment="bottom", fontsize="xx-small")
      ax_f0[i].axvline(lab[i].start[j], color="b", linewidth=1)
      ax_f0[i].axvline(lab[i].end[j], color="b", linewidth=1)
    draw_pitch2(ax_f0[i], pitch[i], lab[i])
  # Draw F0 ratio
  fmin0 = call(pitch[0], "Get quantile", 0.0, 0.0, 0.00, "Hertz (logarithmic)")
  fmax0 = call(pitch[0], "Get quantile", 0.0, 0.0, 1.00, "Hertz (logarithmic)")
  fmin1 = call(pitch[1], "Get quantile", 0.0, 0.0, 0.00, "Hertz (logarithmic)")
  fmax1 = call(pitch[1], "Get quantile", 0.0, 0.0, 1.00, "Hertz (logarithmic)")
  ax_f0[0].axhline(fmin0, color="r", linewidth=1)
  ax_f0[0].axhline(fmax0, color="r", linewidth=1)
  ax_f0[1].axhline(fmin1, color="r", linewidth=1)
  ax_f0[1].axhline(fmax1, color="r", linewidth=1)
  for j in range(lab[0].length):
    m0 = call(pitch[0], "Get quantile", lab[0].start[j], lab[0].end[j], 0.5, "Hertz (logarithmic)")
    m1 = call(pitch[1], "Get quantile", lab[1].start[j], lab[1].end[j], 0.5, "Hertz (logarithmic)")
    ax_f0[0].plot([lab[0].start[j], lab[0].end[j]],[m0,m0], color="g")
    ax_f0[1].plot([lab[1].start[j], lab[1].end[j]],[m1,m1], color="g")
    s0 = GetSlope(pitch[0], lab[0].start[j], lab[0].end[j])
    s1 = GetSlope(pitch[1], lab[1].start[j], lab[1].end[j])
    p0 = (np.log2(m0)-np.log2(fmin0))/(np.log2(fmax0)-np.log2(fmin0))
    ax_f0[0].text(lab[0].end[j], m0, "{:6.3f}".format(p0),fontsize="xx-small")
    ax_f0[0].text(lab[0].end[j], m0, "{:6.3f}".format(s0), verticalalignment="top",fontsize="xx-small")
    p1 = (np.log2(m1)-np.log2(fmin1))/(np.log2(fmax1)-np.log2(fmin1))
    ax_f0[1].text(lab[1].end[j], m1, "{:6.3f}".format(p1),fontsize="xx-small")
    ax_f0[1].text(lab[1].end[j], m1, "{:6.3f}".format(s1), verticalalignment="top",fontsize="xx-small")

  ax_f0[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,_: f'{y:.0f}'))
  ax_f0[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,_: f'{y:.0f}'))
  ax_f0r.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,_: f'{y:.1f}'))
  
  plt.savefig(figname)
  plt.show() # or plt.savefig("figs/sound-f0.png"), or plt.savefig("figs/sound-f0.pdf")
  

  print("Both Channels")
  datplay(snd0.values, snd0.sampling_frequency)
  print("Channel #0 (Left)")
  datplay(snd[0].values, snd[0].sampling_frequency)
  print("Channel #1 (Right)")
  datplay(snd[1].values, snd[1].sampling_frequency)

sns.set()
plt.rcParams['figure.dpi'] = 200
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["axes.labelsize"] = 8