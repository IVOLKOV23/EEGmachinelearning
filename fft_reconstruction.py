
from eeghelpers import *

if __name__ == '__main__':
	# Create a known signal so we can look at how bins impacts the reconstruction from FFT.
	# We want to have confidence that our FFT vector is truly representative of the underlying data.
	"""
	If we ahve 8192 samples for the FFT then we will have:	8192 samples/2=4096 FFT bins
	If our sampling rate is 10 kHz, then the Nyquist-Shannon sampling theorem says that our signal can contain frequency content up to 5 kHz. Then, our frequency bin resolution is:
	5 kHz/4096FFT bins≃1.22 Hz/bin
	Min Bin resolution is just fsamp/N, where fsamp is the input signal's sampling rate and N is the number of FFT points used (sample length).
	We can see from the above that to get smaller FFT bins we can either run a longer FFT (that is, take more samples at the same rate before running the FFT) or decrease our sampling rate.
	"""
	m = MNEDataset(batch_step=.5, batch_window=1)

	samplerate = m.samplerate #todo: Try different samplerates. Make sure
	# nbins=20 #todo: Try different nbins
	# t=np.arange(0,10,1/m.samplerate) #todo: try different durations of time with the same sample rate.
	# #t = np.arange(0, 1, 1 / m.samplerate)
	# f1 = 10
	# f2 = 100
	# f3 = 75
	# #todo: try different functions.
	# sig1 = 10 * np.sin(2. * np.pi * f1 * t)
	# sig2 = 5 * np.sin(2. * np.pi * f2 * t)
	# sig3 = 2 * np.cos(2. * np.pi * f3 * t)
	# temporal = sig1 + sig2 +sig3
	#
	# m.plot_recon([temporal], t,  nbins=nbins)
	# #note that the temporal signal needs to be 2 dimensional.
	# This allows you to do all of the EEG channels at once - then extract the channel after the FFT.

	####now do it for a snippet of EEG
	nbins = 626
	#todo: Try different nbins. Note the reconstruction t decreases as bins decrease.
	# Because reconstruction is periodic but EEG is not.
	# We can use this analysis to determine what length of EEG we can use for FFT.
	# We have to trade of the number of bins (vector size) for lenth of EEG window.
	# Vector size is limited by processing hardware.
	# What is the longest EEG window for nbins = 1000 ?
	# This would mean we have 1000x2 inputs into the Artifical Neural Network (ANN). How bout nbins=500?

	ts = 5
	m = MNEDataset(batch_step=0.5*ts/60, batch_window=ts/60)
	print(f"****************************NOW DOING SOME EEG*****************************")
	data, times, last = m.get_next_batch()
	m.plot_recon(data, times,nbins=nbins,channel=23)

	print("Done")