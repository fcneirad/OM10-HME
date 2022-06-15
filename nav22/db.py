import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
pd.set_option('display.max_columns', None)

class DB(object):
    """
    Reads the lensed images and stores ir in a pandas dataframe

    Parameters
    ----------
    band : string, optional
        Photometric band from which the events will be shown.
        Default is 'r', must be one of the LSST photometric bands.

    threshold: float, optional
        Minimum amplitude of the events in the corresponding photometric band.
        Default is 0.3, must be either 0.3, 0.5 or 1.0.
    """

    
    def __init__(self, band='r', threshold=0.3, verbose=True):

        current_path = os.path.abspath(os.getcwd())
        self.path_to_cdfs = current_path+'/CDFs/'
        self.path_to_lenses = current_path+'/lenses/'
        self.band = band
        self.threshold = threshold
        self.verbose = verbose

        self.setup()

        return

    def setup(self):
        """
        Set initial population of lensed images.
        You can call this method again to reset any constrain
        that you may have applied to the dataset.
        """
        
        self.lenses = pd.read_csv(self.path_to_lenses + 'selected_lenses_abmag_ABCD.csv')
        bands = ['u', 'g', 'r', 'i', 'z', 'y']
        thresholds = [1.0, 0.5, 0.3]

        nevents   = np.load(self.path_to_lenses+'all_events.npy')[:,bands.index(self.band), thresholds.index(self.threshold)]
        nccStrong = np.load(self.path_to_lenses+'strong_events.npy')[:,bands.index(self.band), thresholds.index(self.threshold)]
        nccWeak   = np.load(self.path_to_lenses+'weak_events.npy')[:,bands.index(self.band), thresholds.index(self.threshold)]
        nccSingle = np.load(self.path_to_lenses+'single_events.npy')[:,bands.index(self.band), thresholds.index(self.threshold)]
        nccAll    = nccStrong + nccWeak + nccSingle

        self.lenses['AllEvents'] = nevents
        self.lenses['AllCcross'] = nccAll
        self.lenses['Strong'] = nccStrong
        self.lenses['Weak'] = nccWeak
        self.lenses['Single'] = nccSingle
        
        doubles = int(len(self.lenses[self.lenses['NIMG']==2])/2)
        naked = len(self.lenses[self.lenses['NIMG']==3])/3
        quads = len(self.lenses[self.lenses['NIMG']==4])/4
        
        if self.verbose:
            print('There are a total of {:d} images from {:d} doubles, {:.2f} naked cusps and {:.2f} quads'.format(len(self.lenses), doubles, naked, quads))
        
        return

    def sampleNlenses(self, area=20000):
        """
        Randomly select the appropiate number of lensed images according
        to the expected lensed systems corresponding to a sky area.

        Parameters
        ----------
        area : float, optional
            In deg^2, default is 20000, must be between 0 and 100000.
        """

        lensids = np.unique([int(i.split('_')[0]) for i in self.lenses['ID'].values])
        fraction = area/1e5
        nlenses = int(len(lensids)*fraction)
        selected_lenses = np.random.choice(np.unique(lensids), size=nlenses, replace=False)
        self.sampled_lenses = self.lenses[self.lenses['LENSID'].isin(selected_lenses)]

        return

    def expectedEvents(self):
        """
        Plots the expected number of events in 10 years for
        each lensed image currently in the dataframe.
        """

        bins = np.linspace(0, 3, 50)
        fig, ax = plt.subplots(1, 1, figsize=(12,12))
        ax.set_title('Threshold = ' + str(self.threshold) + '  ' + self.band+'-band', fontsize=24)
        ax.hist(self.lenses[self.lenses['Parity']=='Minimum']['AllEvents'], bins=bins, density=False, alpha=0.5, label='Minimum images')
        ax.hist(self.lenses[self.lenses['Parity']=='Saddle']['AllEvents'], bins=bins, density=False, alpha=0.5, label='Saddle images')
        ax.set_xlabel('Expected no. of events in 10 years', fontsize=24)
        ax.set_ylabel('Number of lensed images', fontsize=24)
        ax.tick_params(axis='both', labelsize=24)
        ax.legend(prop={'size': 24});

        return

    def fractionCaustic(self):
        """
        Plots the fraction of caustic crossing events for
        each lensed image currently in the dataframe.
        """

        bins = np.linspace(0, 1, 50)
        fig, ax = plt.subplots(1, 1, figsize=(12,12))
        ax.set_title('Threshold = ' + str(self.threshold) + '  ' + self.band+'-band', fontsize=24)
        ax.hist(self.lenses[self.lenses['Parity']=='Minimum']['AllCcross']/self.lenses[self.lenses['Parity']=='Minimum']['AllEvents'], bins=bins, density=False, alpha=0.5, label='Minimum images')
        ax.hist(self.lenses[self.lenses['Parity']=='Saddle']['AllCcross']/self.lenses[self.lenses['Parity']=='Saddle']['AllEvents'], bins=bins, density=False, alpha=0.5, label='Saddle images')
        ax.set_xlabel('Fraction of caustic crossing events', fontsize=24)
        ax.set_ylabel('Number of lensed images', fontsize=24)
        ax.tick_params(axis='both', labelsize=24)
        ax.legend(prop={'size': 24});

        return

    def select_images(self, random=True, Nimages=1, parity=None, ids=None):
        """
        Select from the lensed images currently in the dataframe.

        Parameters
        ----------
        random : bool, optional
            If True, randomly select Nimages, default is True.
        Nimages : int, optional
            The number of images to randomly select, default is 1.
        Parity : None or str, optional
            Choose if you want to select only lensed images with
            the specified parity. Can be None, 'Minimum' or 'Saddle',
            default is None. If None, then the lensed images are
            selected from both parities.
        ids : list, array-like of str or None, optional
            If provided random must be set to False. Instead of
            randomly selecting images, you can provide a list.
            Must be the lensed image id, not the lens id (i.e. lens id =
            4060, lensed image id = 4060_A).
        """

        if random:
            if not parity:
                self.images = self.lenses.sample(Nimages)
            elif parity == 'Minimum' or parity == 'Saddle':
                self.images = self.lenses[self.lenses['Parity']==parity].sample(Nimages)
            else:
                raise ValueError('parity has to be either None, "Minimum" or "Saddle"')

        else:
            self.images = self.lenses.loc[ids.index, :]

        return 

    def sample_events(self, sample_durations=True, sample_magnitudes=True, cc_only=False, exclude_cc=False):
        """
        The method select_images must be called before. Generates the
        durations and amplitudes of the events from the selected lensed
        images.

        Parameters
        ----------
        sample_durations : bool, optional
            Default is True, choose if you want to generate the durations
            of the events.
        sample_magnitudes : bool, optional
            Default is True, chooss if you want to generate the amplitudes
            of the events.
        cc_only : bool, optional
            Default is False, if True only caustic crossing events are
            generated.
        exclude_cc : bool, optional
            Default is False, if True only non-caustic crossing events
            are generated.
        """

        self.durations_sampled = sample_durations
        self.magnitudes_sampled = sample_magnitudes
        self.cc = cc_only
        self.exclude_cc = exclude_cc

        try:
            bands = ['u', 'g', 'r', 'i', 'z', 'y']
            index = self.images.index

            if self.cc:
                path_to_events = self.path_to_cdfs + 'caustic_crossings/'
                nevents_strong = self.images['Strong'].values*100000
                nevents_weak   = self.images['Weak'].values*100000
                nevents_single = self.images['Single'].values*100000
                if sample_durations:
                    ## sample duration of events
                    bins = np.arange(0, 3652.5, 20)
                    x = (bins[1:] + bins[:-1]) * 0.5
                    durations_strong = np.load(path_to_events + 'strong/duration_' + str(int(self.threshold*10)).zfill(2) + '.npy', mmap_mode='r')[index,:,bands.index(self.band)]
                    durations_weak   = np.load(path_to_events + 'weak/duration_'   + str(int(self.threshold*10)).zfill(2) + '.npy', mmap_mode='r')[index,:,bands.index(self.band)]
                    durations_single = np.load(path_to_events + 'single/duration_' + str(int(self.threshold*10)).zfill(2) + '.npy', mmap_mode='r')[index,:,bands.index(self.band)]
                    interpolators_durations_strong = [interp1d(durations_strong[i,:], x, bounds_error=False, fill_value=(x[0], x[-1])) for i in range(len(index))]
                    interpolators_durations_weak   = [interp1d(durations_weak[i,:],   x, bounds_error=False, fill_value=(x[0], x[-1])) for i in range(len(index))]
                    interpolators_durations_single = [interp1d(durations_single[i,:], x, bounds_error=False, fill_value=(x[0], x[-1])) for i in range(len(index))]
                    F_inv_durations_strong = lambda x: np.asarray([i(np.random.uniform(size=int(x[j]))) for j,i in enumerate(interpolators_durations_strong)], dtype=object)
                    F_inv_durations_weak   = lambda x: np.asarray([i(np.random.uniform(size=int(x[j]))) for j,i in enumerate(interpolators_durations_weak)], dtype=object)
                    F_inv_durations_single = lambda x: np.asarray([i(np.random.uniform(size=int(x[j]))) for j,i in enumerate(interpolators_durations_single)], dtype=object)
                    self.durations_strong = F_inv_durations_strong(nevents_strong)
                    self.durations_weak   = F_inv_durations_weak(nevents_weak)
                    self.durations_single = F_inv_durations_single(nevents_single)

                if sample_magnitudes:
                    ## sample magnitude of events
                    bins = np.arange(0, 5, 0.05)
                    x = (bins[1:] + bins[:-1]) * 0.5
                    magnitudes_strong = np.load(path_to_events + 'strong/magnitude_' + str(int(self.threshold*10)).zfill(2) + '.npy', mmap_mode='r')[index,:,bands.index(self.band)]
                    magnitudes_weak   = np.load(path_to_events + 'weak/magnitude_'   + str(int(self.threshold*10)).zfill(2) + '.npy', mmap_mode='r')[index,:,bands.index(self.band)]
                    magnitudes_single = np.load(path_to_events + 'single/magnitude_' + str(int(self.threshold*10)).zfill(2) + '.npy', mmap_mode='r')[index,:,bands.index(self.band)]
                    interpolators_magnitudes_strong = [interp1d(magnitudes_strong[i,:], x, bounds_error=False, fill_value=(x[0], x[-1])) for i in range(len(index))]
                    interpolators_magnitudes_weak   = [interp1d(magnitudes_weak[i,:],   x, bounds_error=False, fill_value=(x[0], x[-1])) for i in range(len(index))]
                    interpolators_magnitudes_single = [interp1d(magnitudes_single[i,:], x, bounds_error=False, fill_value=(x[0], x[-1])) for i in range(len(index))]
                    F_inv_magnitudes_strong = lambda x: np.asarray([i(np.random.uniform(size=int(x[j]))) for j,i in enumerate(interpolators_magnitudes_strong)], dtype=object)
                    F_inv_magnitudes_weak   = lambda x: np.asarray([i(np.random.uniform(size=int(x[j]))) for j,i in enumerate(interpolators_magnitudes_weak)], dtype=object)
                    F_inv_magnitudes_single = lambda x: np.asarray([i(np.random.uniform(size=int(x[j]))) for j,i in enumerate(interpolators_magnitudes_single)], dtype=object)
                    self.magnitudes_strong = F_inv_magnitudes_strong(nevents_strong)
                    self.magnitudes_weak   = F_inv_magnitudes_weak(nevents_weak)
                    self.magnitudes_single = F_inv_magnitudes_single(nevents_single)

                    
            else:
                if self.exclude_cc:
                    path_to_cc_events  = self.path_to_cdfs + 'caustic_crossings/'
                    path_to_all_events = self.path_to_cdfs + 'all_events/'
                    nevents_all = (self.images['AllEvents'].values*100000)[:,np.newaxis]
                    nevents_strong = (self.images['Strong'].values*100000)[:,np.newaxis]
                    nevents_weak   = (self.images['Weak'].values*100000)[:,np.newaxis]
                    nevents_single = (self.images['Single'].values*100000)[:,np.newaxis]
                    nevents_ncc = nevents_all - nevents_strong - nevents_weak - nevents_single
                    if sample_durations:
                        ## sample duration of events
                        bins = np.arange(0, 3652.5, 20)
                        x = (bins[1:] + bins[:-1]) * 0.5
                        durations_all  = np.nan_to_num(np.load(path_to_all_events + 'duration_' + str(int(self.threshold*10)).zfill(2) + '.npy', mmap_mode='r')[index,:,bands.index(self.band)])
                        durations_strong = np.nan_to_num(np.load(path_to_cc_events + 'strong/duration_' + str(int(self.threshold*10)).zfill(2) + '.npy', mmap_mode='r')[index,:,bands.index(self.band)])
                        durations_weak   = np.nan_to_num(np.load(path_to_cc_events + 'weak/duration_'   + str(int(self.threshold*10)).zfill(2) + '.npy', mmap_mode='r')[index,:,bands.index(self.band)])
                        durations_single = np.nan_to_num(np.load(path_to_cc_events + 'single/duration_' + str(int(self.threshold*10)).zfill(2) + '.npy', mmap_mode='r')[index,:,bands.index(self.band)])
                        durations = durations_all * nevents_all - durations_strong * nevents_strong - durations_weak * nevents_weak - durations_single * nevents_single
                        durations = durations/np.max(durations, axis=1)[:,np.newaxis]
                        interpolators_durations = [interp1d(durations[i,:], x, bounds_error=False, fill_value=(x[0], x[-1])) for i in range(len(index))]
                        F_inv_durations = lambda x: np.asarray([i(np.random.uniform(size=int(x[j]))) for j,i in enumerate(interpolators_durations)], dtype=object)
                        self.durations = F_inv_durations(nevents_ncc)

                    if sample_magnitudes:
                        ## sample magnitude of events
                        bins = np.arange(0, 5, 0.05)
                        x = (bins[1:] + bins[:-1]) * 0.5
                        magnitudes_all  = np.nan_to_num(np.load(path_to_all_events + 'magnitude_' + str(int(self.threshold*10)).zfill(2) + '.npy', mmap_mode='r')[index,:,bands.index(self.band)])
                        magnitudes_strong = np.nan_to_num(np.load(path_to_cc_events + 'strong/magnitude_' + str(int(self.threshold*10)).zfill(2) + '.npy', mmap_mode='r')[index,:,bands.index(self.band)])
                        magnitudes_weak   = np.nan_to_num(np.load(path_to_cc_events + 'weak/magnitude_'   + str(int(self.threshold*10)).zfill(2) + '.npy', mmap_mode='r')[index,:,bands.index(self.band)])
                        magnitudes_single = np.nan_to_num(np.load(path_to_cc_events + 'single/magnitude_' + str(int(self.threshold*10)).zfill(2) + '.npy', mmap_mode='r')[index,:,bands.index(self.band)])
                        magnitudes = magnitudes_all * nevents_all - magnitudes_strong * nevents_strong - magnitudes_weak * nevents_weak - magnitudes_single * nevents_single
                        magnitudes = magnitudes/np.max(magnitudes, axis=1)[:,np.newaxis]
                        interpolators_magnitudes = [interp1d(magnitudes[i,:], x, bounds_error=False, fill_value=(x[0], x[-1])) for i in range(len(index))]
                        F_inv_magnitudes = lambda x: np.asarray([i(np.random.uniform(size=int(x[j]))) for j,i in enumerate(interpolators_magnitudes)], dtype=object)
                        self.magnitudes = F_inv_magnitudes(nevents_ncc)


                else:
                    path_to_events = self.path_to_cdfs + 'all_events/'
                    nevents_all   = self.images['AllEvents'].values*100000
                    if sample_durations:
                        ## sample duration of events
                        bins = np.arange(0, 3652.5, 20)
                        x = (bins[1:] + bins[:-1]) * 0.5
                        durations  = np.load(path_to_events + 'duration_' + str(int(self.threshold*10)).zfill(2) + '.npy', mmap_mode='r')[index,:,bands.index(self.band)]
                        interpolators_durations = [interp1d(durations[i,:], x, bounds_error=False, fill_value=(x[0], x[-1])) for i in range(len(index))]
                        F_inv_durations = lambda x: np.asarray([i(np.random.uniform(size=int(x[j]))) for j,i in enumerate(interpolators_durations)], dtype=object)
                        self.durations = F_inv_durations(nevents_all)

                    if sample_magnitudes:
                        ## sample magnitude of events
                        bins = np.arange(0, 5, 0.05)
                        x = (bins[1:] + bins[:-1]) * 0.5
                        magnitudes = np.load(path_to_events + 'magnitude_' + str(int(self.threshold*10)).zfill(2) + '.npy', mmap_mode='r')[index,:,bands.index(self.band)]
                        interpolators_magnitudes = [interp1d(magnitudes[i,:], x, bounds_error=False, fill_value=(x[0], x[-1])) for i in range(len(index))]
                        F_inv_magnitudes = lambda x: np.asarray([i(np.random.uniform(size=int(x[j]))) for j,i in enumerate(interpolators_magnitudes)], dtype=object)
                        self.magnitudes = F_inv_magnitudes(nevents_all)

        except:
            raise ClassException('You need to select at least one image using the select_images method first.')

        return

    def plot_selected(self):
        """
        The method sample_events must be called before. Generates the plots
        for all the properties of the events generated from the selected
        lensed images.
        """

        labels = [str(int(i['LENSID'])) + '_' + i['IMG'] for _,i in self.images.iterrows()]

        try:
            if self.cc:
                if self.durations_sampled:
                    bins_durations = np.linspace(0, 3652.5+80, 50)
                    fig, axs = plt.subplots(1, len(labels), figsize=(12*len(labels),12), sharey=True)
                    for i, label in enumerate(labels):
                        axs[i].set_title(label, y=0.95, fontsize=24)
                        axs[i].hist(self.durations_strong[i], bins=bins_durations, density=True, color='b', alpha=0.5, label='strong')
                        axs[i].hist(self.durations_weak[i], bins=bins_durations, density=True, color='r', alpha=0.5, label='weak')
                        axs[i].hist(self.durations_single[i], bins=bins_durations, density=True, color='g', alpha=0.5, label='single')
                        axs[i].set_xlabel('Duration [days]', fontsize=24)
                        axs[i].set_ylabel('PDF', fontsize=24)
                        axs[i].tick_params(axis='both', labelsize=24)
                        axs[i].legend(prop={'size': 24});

                if self.magnitudes_sampled:
                    bins_magnitudes = np.linspace(0, 3.5, 35)
                    fig, axs = plt.subplots(1, len(labels), figsize=(12*len(labels),12), sharey=True)
                    for i, label in enumerate(labels):
                        axs[i].set_title(label, y=0.95, fontsize=24)
                        axs[i].hist(self.magnitudes_strong[i], bins=bins_magnitudes, density=True, color='b', alpha=0.5, label='strong')
                        axs[i].hist(self.magnitudes_weak[i], bins=bins_magnitudes, density=True, color='r', alpha=0.5, label='weak')
                        axs[i].hist(self.magnitudes_single[i], bins=bins_magnitudes, density=True, color='g', alpha=0.5, label='single')
                        axs[i].set_xlabel('$\Delta$mag', fontsize=24)
                        axs[i].set_ylabel('PDF', fontsize=24)
                        axs[i].tick_params(axis='both', labelsize=24)
                        axs[i].legend(prop={'size': 24});



            else:
                if self.durations_sampled:
                    bins_durations = np.linspace(0, 3652.5+80, 50)
                    fig, ax = plt.subplots(1, 1, figsize=(12,12))
                    cycler = ax._get_lines.prop_cycler
                    for i, label in enumerate(labels):
                        ax.hist(self.durations[i], bins_durations, fill=False, edgecolor=next(cycler)['color'], histtype='stepfilled', density=True, label=label)
                    ax.legend()
                if self.magnitudes_sampled:
                    bins_magnitudes = np.linspace(0, 3.5, 35)
                    fig, ax = plt.subplots(1, 1, figsize=(12,12))
                    cycler = ax._get_lines.prop_cycler
                    for i, label in enumerate(labels):
                        ax.hist(self.magnitudes[i], bins_magnitudes, fill=False, edgecolor=next(cycler)['color'], histtype='stepfilled', density=True, label=label)
                    ax.legend()
        except:
            raise ClassException('You need to fetch the events using the method fetch_events.')

        return

    def plot_all(self, plot_durations=True, plot_magnitudes=True, cc_only=False, exclude_cc=False):
        """
        Generates the plots of the desired event properties for all the lensed images that
        are currently in the dataframe. The events are classified in the parity of the image
        from which they belong to. It is not required to run the sample_events method, this
        function does it automatically.

        Parameters
        ----------
        plot_durations : bool, optional
            Choose if you want to plot the durations of the events, default is True.
        plot_magnitudes : bool, optional
            Choose if you want to plot the amplitudes of the events, default is True.
        cc_only : bool, optional
            Default is False, if True only caustic crossing events are plotted.
        exclude_cc : bool, optional
            Default is False, if True only non-caustic crossing events are plotted.
        """

        self.cc = cc_only
        self.exclude_cc = exclude_cc

        ids_minima = self.lenses[self.lenses['Parity']=='Minimum']['ID']
        ids_saddle = self.lenses[self.lenses['Parity']=='Saddle']['ID']

        if self.cc:
            self.select_images(random=False, ids=ids_minima)
            self.sample_events(cc_only=True)
            durations_minima_strong  = np.hstack(self.durations_strong)
            durations_minima_weak    = np.hstack(self.durations_weak)
            durations_minima_single  = np.hstack(self.durations_single)
            magnitudes_minima_strong = np.hstack(self.magnitudes_strong)
            magnitudes_minima_weak   = np.hstack(self.magnitudes_weak)
            magnitudes_minima_single = np.hstack(self.magnitudes_single)
            self.select_images(random=False, ids=ids_saddle)
            self.sample_events(cc_only=True)
            durations_saddle_strong  = np.hstack(self.durations_strong)
            durations_saddle_weak    = np.hstack(self.durations_weak)
            durations_saddle_single  = np.hstack(self.durations_single)
            magnitudes_saddle_strong = np.hstack(self.magnitudes_strong)
            magnitudes_saddle_weak   = np.hstack(self.magnitudes_weak)
            magnitudes_saddle_single = np.hstack(self.magnitudes_single)

            if plot_durations:
                bins_durations = np.linspace(-40, 3652.5+80, 50)
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,24), sharex=True)
                ax1.set_title('Duration of caustic crossing events\nThreshold = ' + str(self.threshold) + '  ' + self.band+'-band', fontsize=24)
                fig.subplots_adjust(hspace=0)
                ax1.hist(durations_minima_strong, bins=bins_durations, density=True, color='b', alpha=0.5, label='strong')
                ax1.hist(durations_minima_weak, bins=bins_durations, density=True, color='r', alpha=0.5, label='weak')
                ax1.hist(durations_minima_single, bins=bins_durations, density=True, color='g', alpha=0.5, label='single')
                ax1.text(0.5, 0.95, 'Minimum images', ha='center', transform=ax1.transAxes, fontsize=24)
                ax1.set_ylabel('PDF', fontsize=24)
                ax1.tick_params(axis='both', labelsize=24)
                ax1.legend(prop={'size': 24})

                ax2.hist(durations_saddle_strong, bins=bins_durations, density=True, color='b', alpha=0.5)
                ax2.hist(durations_saddle_weak, bins=bins_durations, density=True, color='r', alpha=0.5)
                ax2.hist(durations_saddle_single, bins=bins_durations, density=True, color='g', alpha=0.5)
                ax2.text(0.5, 0.95, 'Saddle images', ha='center', transform=ax2.transAxes, fontsize=24)
                ax2.set_xlabel('Duration [days]', fontsize=24)
                ax2.set_ylabel('PDF', fontsize=24)
                ax2.tick_params(axis='both', labelsize=24);
            if plot_magnitudes:
                bins_magnitudes = np.arange(0, 3.5, 0.05)
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,24), sharex=True)
                ax1.set_title('Amplitude of caustic crossing events\nThreshold = ' + str(self.threshold) + '  ' + self.band+'-band', fontsize=24)
                fig.subplots_adjust(hspace=0)
                ax1.hist(magnitudes_minima_strong, bins=bins_magnitudes, density=True, color='b', alpha=0.5, label='strong')
                ax1.hist(magnitudes_minima_weak, bins=bins_magnitudes, density=True, color='r', alpha=0.5, label='weak')
                ax1.hist(magnitudes_minima_single, bins=bins_magnitudes, density=True, color='g', alpha=0.5, label='single')
                ax1.text(0.5, 0.95, 'Minimum images', ha='center', transform=ax1.transAxes, fontsize=24)
                ax1.set_ylabel('PDF', fontsize=24)
                ax1.tick_params(axis='both', labelsize=24)
                ax1.legend(prop={'size': 24})

                ax2.hist(magnitudes_saddle_strong, bins=bins_magnitudes, density=True, color='b', alpha=0.5)
                ax2.hist(magnitudes_saddle_weak, bins=bins_magnitudes, density=True, color='r', alpha=0.5)
                ax2.hist(magnitudes_saddle_single, bins=bins_magnitudes, density=True, color='g', alpha=0.5)
                ax2.text(0.5, 0.95, 'Saddle images', ha='center', transform=ax2.transAxes, fontsize=24)
                ax2.set_xlabel('$\Delta$mag', fontsize=24)
                ax2.set_ylabel('PDF', fontsize=24)
                ax2.tick_params(axis='both', labelsize=24);

        else:
            self.select_images(random=False, ids=ids_minima)
            if self.exclude_cc:
                self.sample_events(exclude_cc=self.exclude_cc)
            else:
                self.sample_events()
            durations_minima = np.hstack(self.durations)
            magnitudes_minima = np.hstack(self.magnitudes)
            self.select_images(random=False, ids=ids_saddle)
            if self.exclude_cc:
                self.sample_events(exclude_cc=self.exclude_cc)
            else:
                self.sample_events()
            durations_saddle = np.hstack(self.durations)
            magnitudes_saddle = np.hstack(self.magnitudes)
            
            if plot_durations:
                bins_durations = np.linspace(-40, 3652.5+80, 50)
                fig, ax = plt.subplots(1, 1, figsize=(12,12))
                if self.exclude_cc:
                    ax.set_title('Duration of all non-caustic crossing HMEs\nThreshold = ' + str(self.threshold) + '  ' + self.band+'-band', fontsize=24)
                else:
                    ax.set_title('Duration of all HMEs\nThreshold = ' + str(self.threshold) + '  ' + self.band+'-band', fontsize=24)
                ax.hist(durations_minima, bins=bins_durations, density=True, alpha=0.5, label='Minimum images')
                ax.hist(durations_saddle, bins=bins_durations, density=True, alpha=0.5, label='Saddle images')
                ax.set_xlabel('Duration [days]', fontsize=24)
                ax.set_ylabel('PDF', fontsize=24)
                ax.tick_params(axis='both', labelsize=24)
                ax.legend(prop={'size': 24});
            if plot_magnitudes:
                bins_magnitudes = np.arange(0, 3.5, 0.05)
                fig, ax = plt.subplots(1, 1, figsize=(12,12))
                if self.exclude_cc:
                    ax.set_title('Amplitude of all non-caustic crossing HMEs\nThreshold = ' + str(self.threshold) + '  ' + self.band+'-band', fontsize=24)
                else:
                    ax.set_title('Amplitude of all HMEs\nThreshold = ' + str(self.threshold) + '  ' + self.band+'-band', fontsize=24)
                ax.hist(magnitudes_minima, bins=bins_magnitudes, density=True, alpha=0.5, label='Minimum images')
                ax.hist(magnitudes_saddle, bins=bins_magnitudes, density=True, alpha=0.5, label='Saddle images')
                ax.set_xlabel('$\Delta$mag', fontsize=24)
                ax.set_ylabel('PDF', fontsize=24)
                ax.tick_params(axis='both', labelsize=24)
                ax.legend(prop={'size': 24});



        return


class ClassException(Exception):
    pass
