import mne
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv
import scipy


class MNEDataset:
	def __init__(self, batch_step=.5, batch_window=1, samplerate=250):
		self.initMNE(batch_step=batch_step, batch_window=batch_window, samplerate=samplerate)

	def initMNE(self, batch_step=.5, batch_window=1, samplerate=250):
		basefolder = "./EEG"
		edfs = glob.glob(os.path.join(basefolder, "*.edf"))
		data = mne.io.read_raw_edf(edfs[0])
		self.last_sample = data.last_samp

		raw_data = data.get_data()

		# you can get the metadata included in the file and a list of all channels:
		self.info = data.info  # https://mne.tools/stable/generated/mne.io.read_raw_edf.html
		self.channels = data.ch_names

		# figure out a sample rate
		samplerate2 = self.info['sfreq']  # this is not always set
		self.samplerate = samplerate
		if samplerate2 != samplerate:
			print("set sample rate is different from info sample rate")

		# info = mne.create_info(ch_names=channels, sfreq=sr,ch_types="eeg").set_montage(montage)
		# https://www.researchgate.net/profile/Heidelise_Als/publication/228064755/figure/fig1/AS:203120069091344@1425439006780/Standard-EEG-electrode-names-and-positions-Head-in-vertex-view-nose-above-left-ear-to.png
		# data.plot()

		self.layout = mne.channels.read_layout('biosemi')
		self.edffiles = edfs
		self.edfdata = data
		self.rawtime = data.times
		self.batch_step = batch_step  # this how much we increment step in minutes.
		self.batch_window = batch_window  # this is how wide the window is in minutes.
		self.idx = 0  # this is how many time steps have been completed.

		self.rawdata = raw_data[3]  # channel specification
		# self.meanEEGPower = np.mean(self.rawdata)
		# self.maxpower = np.max(self.rawdata)
		# self.minpower = np.min(self.rawdata)

	def get_next_batch(self):
		# gets the next time step of data from the EEG.
		start_time = self.idx * self.batch_step
		duration = self.batch_window
		data, times, last = self.get_data_time(start_time, duration)

		if last:
			self.idx = 0
		else:
			self.idx = self.idx + 1

		return data, times, last

	def get_data_time(self, start_time_mins, duration_mins):
		# start and finish sample indices for the signal section
		sample_S = int(60 * start_time_mins * self.samplerate)
		sample_finish = int(60 * (start_time_mins + duration_mins) * self.samplerate)

		last = False
		if sample_finish > self.last_sample:
			# get data and time for the last interval from the raw data
			raw_data, times = self.edfdata.get_data(start=sample_S, stop=self.last_sample, return_times=True)
			last = True
		else:
			# raw_data,times = self.edfdata.get_data_samples(start=sample_S,stop=sample_finish,return_times=True)
			# get the data values from the EEG
			raw_data, times = self.edfdata.get_data(start=sample_S, stop=sample_finish, return_times=True)

		return raw_data, times, last

	def get_data_samples(self, start, finish, return_times=True):
		# samples from the data
		raw_data, times = self.edfdata.get_data(start=start, stop=finish, return_times=return_times)
		return raw_data, times


if __name__ == '__main__':
	m = MNEDataset(batch_step=.25, batch_window=1)

	# the first minute of the data
	# to initialise "last" I guess??
	# 30 sec window and 15 sec step
	# why? it just messes the whole thing up
	# data, times, last = m.get_next_batch()
	# freq = 1 / times[1]

	###Calling functions directly
	# one second worth of samples
	showdata_s, time_s = m.get_data_samples(0, m.samplerate)  # this should be 1 seconds worth of samples
	showdata_t, time2_t, last = m.get_data_time(0, 1 / 60)  # this is 1 second worth o samples should be same as above

	###Cycling through the data.
	counter = 0
	while not last:
		# again starts from the first minute
		# get the first batch of data
		data, times, last = m.get_next_batch()

		# get channel 3 data only
		datanp = data[2]
		datanp = np.array(datanp)

		# get fft of the batch
		datafft = scipy.fft.rfft(datanp)

		# get the freq axis for the graph
		xf = scipy.fft.rfftfreq(len(datanp))

		####
		# 1. Simple Reconstruction
		yrec = scipy.fft.irfft(datafft)

		fig, (ax1, ax2, ax3) = plt.subplots(3)
		ax1.plot(datanp)
		ax1.set_title('Original Data')
		ax1.set_xlabel('Time (sec)')
		ax1.set_ylabel('Amplitude (a.u.)')
		ax2.plot(yrec)
		ax2.set_title('Reconstructed')
		ax2.set_xlabel('Time (sec)')
		ax2.set_ylabel('Amplitude (a.u.)')
		ax3.plot(yrec - datanp)
		ax3.set_title('Reconstruction Error')
		ax3.set_xlabel('Time (sec)')
		ax3.set_ylabel('Amplitude (a.u.)')
		fig.tight_layout(pad=1.0)

		###
		# 2. Magnitude and Phase
		# phase
		phase = np.angle(datafft)

		# magnitude + normalisation
		mag = np.abs(datafft)/len(datafft)

		# convert back to complex
		recfft = mag*len(datafft)*np.exp(1j*phase)

		# back to time domain
		yrec2 = scipy.fft.irfft(recfft)

		# plotting
		fig, (ax1, ax2) = plt.subplots(2)
		ax1.plot(xf, mag)
		ax1.set_title('Magnitude')
		ax1.set_xlabel('Freq (Hz)')
		ax1.set_ylabel('Magnitude (a.u.)')
		ax2.plot(xf, phase)
		ax2.set_title('Phase')
		ax2.set_xlabel('Freq (Hz)')
		ax2.set_ylabel('Phase (a.u.)')
		fig.tight_layout(pad=1.0)

		# Reconstruct back to Time domain again.
		# plotting
		fig, (ax1, ax2, ax3) = plt.subplots(3)
		ax1.plot(yrec2)
		ax1.set_title('Mag and Phase Reconstruction')
		ax1.set_xlabel('Time (sec)')
		ax1.set_ylabel('Amplitude (a.u.)')
		ax2.plot(yrec2 - datanp)
		ax2.set_title('Reconstruction Error 2')
		ax2.set_xlabel('Time (sec)')
		ax2.set_ylabel('Amplitude (a.u.)')
		ax3.plot(yrec - datanp)
		ax3.set_title('Reconstruction Error')
		ax3.set_xlabel('Time (sec)')
		ax3.set_ylabel('Amplitude (a.u.)')
		fig.tight_layout(pad=1.0)
		plt.show()

		###
		# 3. Create N bins of frequency for 2 - digitize  scipy library
		# t->fft->complex-> Mag/Phase -> nbins -> complex -> ifft -> t
		# Reconstruct and play with N to see the impact on reconstruction loss
		# You want to have some predetermined acceptable loss of reconstruction dependent on the biomarker
		# you hypothesise exists.

		# plotting
		# plt.plot(xf, datafft) - fft plot
		# plt.show()

		counter += 1

		if last:
			print(f"{times[-1] / 60}s")
			print(f"{counter} steps")

	# does it plot just one second??
	# convert ot actual array
	# showdataNP = np.array(showdata_s)
	# image creation
	# showdataNP = cv.normalize(showdataNP, showdata_s, 0, 255, cv.NORM_MINMAX)
	# plt.imshow(showdataNP, cmap="gray", interpolation=None)
	# img = Image.fromarray(showdataNP, 'L')

	# fft plot
	# convert fft back to regular signal
	# signal = scipy.fft.ifft(fftvector)
	# signal = np.real(signal)

	# img.show()
	# plt.show()

	# Use the plotting tools to show the eeg
	# mne.viz.plot_raw(m.edfdata)

	print("Done")
