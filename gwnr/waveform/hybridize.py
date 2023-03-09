''' 
Master function to hybridise any complex timeseries using the 'frequency' as a user input, 
specifically to be used for gravitational waveform hybridisation, fine-tuned for a single mode.

Developed by: Kartikey Sharma

'''



def perform_hybridisation(inspiral, merger_ringdown, frq_attach, frq_width, tol = 0.1, no_sp = 4):
    
    
    timeaxis_insp = np.arange(len(inspiral))*dt
    timeaxis_mr = np.arange(len(merger_ringdown))*dt
    
    
    ''' Computing waveform attributes: amplitude, phase, frequency (no interpolation) '''

    def compute_amplitude(waveform):
        amplitude = np.abs(waveform)
        return amplitude
    
    def compute_phase(waveform): 
        phase = np.unwrap(-np.angle(waveform))
        return phase
    
    def compute_frequency(phase):
        frequency = np.diff(phase)/(dt*2*np.pi)
        return frequency
    
    
    # frqisco = 1/(6*np.sqrt(6)*2*np.pi*(mass1+mass2))*(c3_by_G) # we will not have m1 and m2 as input 
    phase_insp = compute_phase(inspiral)
    frq_insp = compute_frequency(phase_insp)
    amp_insp = compute_amplitude(inspiral)

    phase_mr = compute_phase(merger_ringdown)
    frq_mr = compute_frequency(phase_mr)
    amp_mr = compute_amplitude(merger_ringdown)

    ''' first we need to find the attachment region, based on the frequency '''
    
    def find_value_location_in_series(frq_timeseries, frq_desired): # More outside the main fn
    
        if frq_desired < frq_min:
            raise Exception('Desired frequency out of bounds, lower than min frequency')

        if frq_desired > np.max(frq_timeseries):
            raise Exception('Desired frequency out of bounds, higher than max frequency')

        # tol = 0.1 # error tolerance in percentage, fails at around 0.0001% (user input, default = 0.1)

        i1_tentative = np.logical_and(frq_timeseries >= (1-tol)*frq_desired, frq_timeseries <= (1+tol)*frq_desired) 
        ''' i1_tentative gives a boolean numpy array (length same as frq_timeseries) 
        where the frq values are within the tolerance wrt desired frq, 
        there will be multiple such values '''

        where_i1 = np.where(i1_tentative)
        ''' where_i1 is a tuple that holds a small array of the index values of frq_timeseries,
        corresponding to the indices where i1_tentative is TRUE '''

        cost_i1 = np.square(frq_timeseries[where_i1] - frq_desired)
        ''' cost_i1 is a cost fn which minimised the square of the difference between the frq values 
        at the indices provided by where_i1, and the desired frq, this will have the length same as the array held by the tuple where_i1 '''

        i1_forsure = np.where(cost_i1 == np.min(cost_i1))
        ''' i_forsure is a tuple that holds an array which holds the numerical value of the index where cost_i1 is minimum '''

        i1 = where_i1[0][i1_forsure[0][0]] 
        ''' i1 is an integer, which is the numerical value of the index value where frq_desired is located, 
        it is extracted from the tuple where_i1'''
        return i1
    
    
    
    t1_index_insp = find_value_location_in_series(frq_insp, frq_attach - frq_width/2)
    t2_index_insp = find_value_location_in_series(frq_insp, frq_attach + frq_width/2)


    t1_index_mr_tent = find_value_location_in_series(frq_mr, frq_attach - frq_width/2)
    # t2_index_mr_tent = find_i1(frq_mr, frq_attach + frq_width/2) 

    # another way to define t2_index_mr_tent is through number of points in the inspiral window    
    t2_index_mr_tent = t1_index_mr_tent + (t2_index_insp-t1_index_insp) 
    ''' 
        Theoretically, we NEED a timeshift to align the waveforms in frequency. 
        Instead of shifting one of the two waveforms for alignment, we are defining
        the time such that the frequencies are pre-aligned to the best of the 
        discrete interval errors. That is: 
            deltaT (timeshift) = t1_index_insp - t1_index_mr_tent
        The mathematical way is to optimise the difference in frequencies over the matching 
        region and using that to determine deltaT, hence arriving at t1_index_mr. 

    '''

    no_cycles = (phase_insp[t1_index_insp] - phase_insp[t2_index_insp])/(2*np.pi)


    ''' After alignment, need to find corresponding indices for merger_ringdown as well '''
    
    
    ''' Alignment: '''
    
    # Defining the Comb
    
    def get_comb(t1_index_insp, t2_index_insp, no_sp):

        d = int((t2_index_insp-t1_index_insp)/(no_sp-1))

        count1 = np.empty(no_sp-1)
        for kk in range(no_sp-1):
            count1[kk] = kk*d

        samples = np.append(count1, t2_index_insp-t1_index_insp-1).astype(int)
        return samples

    sample_indices_insp = get_comb(t1_index_insp, t2_index_insp, no_sp = 4)
    sample_indices_mr = get_comb(t1_index_mr_tent, t2_index_mr_tent, no_sp = 4)



    ''' Enter alignment code here '''

    ''' NOT USING TIMESHIFTS IN THIS VERSION OF THE ALGORITHM, HENCE INTERPOLANT NOT NEEDED '''

    ''' Need interpolated frequency evaluated on the time axis for timeshifts '''

    # interp_phase_insp = UnivariateSpline(timeaxis_insp[t1_index_insp:t2_index_insp], phase_insp[t1_index_insp:t2_index_insp], k = 4)
    # interp_phase_mr = UnivariateSpline(timeaxis_mr[t1_index_mr_tent:t2_index_mr_tent], phase_mr[t1_index_mr_tent:t2_index_mr_tent], k = 4)

    # interp_frq_insp = interp_phase_insp.derivative()(timeaxis_insp[t1_index_insp:t2_index_insp])/(2*np.pi)
    # interp_frq_mr = interp_phase_mr.derivative()(timeaxis_mr[t1_index_mr_tent:t2_index_mr_tent])/(2*np.pi)





    ''' alignment using corrective phase addition '''

    def mismatch_discrete(w1, w2): # More outside the main fn
        w1_d = w1[sample_indices_insp]
        w2_d = w2[sample_indices_mr] # can't give the same comb to w2
        w1sq = np.square(np.abs(w1_d))
        # w2sq = np.square(np.abs(w2_d))
        diff = np.abs(w1_d - w2_d)
        diffsq = np.square(diff)
        mm = 0.5*(np.sum(diffsq)/np.sum(w1sq))
        return mm
    


    def align_in_phase(inspiral, merger_ringdown):
    # Function alignes the two waveforms using the phase, optimised over the attachment region

        def optfn_ph(phaseshift_correction):
            phase_corrected_mr = merger_ringdown*np.exp(1j*phaseshift_correction)
            m = mismatch_discrete(inspiral[t1_index_insp:t2_index_insp], phase_corrected_mr[t1_index_mr_tent:t2_index_mr_tent])
            return m

        phase_optimizer = scipy.optimize.minimize(optfn_ph, 0)
        phaseshift_required_for_alignment = phase_optimizer.x

        aligned_merger_ringdown = merger_ringdown*np.exp(1j*phaseshift_required_for_alignment)

        return aligned_merger_ringdown, phaseshift_required_for_alignment 


    merger_ringdown_aligned, phasecorr = align_in_phase(inspiral, merger_ringdown)
    amp_mr_aligned = compute_amplitude(merger_ringdown_aligned)
    phase_mr_aligned = compute_phase(merger_ringdown_aligned)
    frq_mr_aligned = compute_frequency(phase_mr_aligned) # it would be same as frq_mr as the corrected phase factor will be canceled in the derivative


    ''' Performing attachment using the blending function '''

    def blend_series(x1, x2, t1_index_insp, t2_index_insp, t1_index_mr_tent, t2_index_mr_tent): # More outside the main fn

        assert t1_index_mr_tent - t2_index_mr_tent == t1_index_insp - t2_index_insp, "Inconsistent indices passed to blending function"
        
        # blending fn is an array
        blfn_var = np.arange(t1_index_insp, t2_index_insp)
        tau = np.square((np.sin((np.pi/2)*(blfn_var-t1_index_insp)/(t2_index_insp-t1_index_insp))))

        x_hyb = (1-tau)*x1[t1_index_insp:t2_index_insp] + tau*x2[t1_index_mr_tent:t2_index_mr_tent]
        return x_hyb


    amp_hyb_window = blend_series(amp_insp, amp_mr_aligned, t1_index_insp, t2_index_insp, t1_index_mr_tent, t2_index_mr_tent)
    frq_hyb_window = blend_series(frq_insp, frq_mr_aligned, t1_index_insp, t2_index_insp, t1_index_mr_tent, t2_index_mr_tent)

    ''' Integrating frq_hyb to obtain phase_hyb and removing discontinuities, 
        compiling amp_hyb and phase_hyb to obtain the hybrid waveform.     ''' 



    phase_hyb_window = (2*np.pi)*cumulative_trapezoid(frq_hyb_window, dx = dt) # Length of this will be one point shorter than frq_hyb_window

    ''' Right now the phase is integrated only inside the hybrid window, 
    need to add constants to preserve phase continuity and compile full IMR phase '''

    def remove_phase_discontinuity(phase_insp, phase_hyb_window, phase_mr_aligned):
        delta1 = phase_insp[t1_index_insp] - phase_hyb_window[0]
        phase_hyb_1 = np.append(phase_insp[:t1_index_insp], phase_hyb_window + delta1)
        delta2 = phase_hyb_1[t2_index_insp - 1] - phase_mr_aligned[t2_index_mr_tent - 1]
        phase_hyb_2 = np.append(phase_hyb_1[:t2_index_insp - 1], phase_mr_aligned[t2_index_mr_tent - 1:] + delta2)
        return phase_hyb_2

    phase_hyb_full = remove_phase_discontinuity(phase_insp, phase_hyb_window, phase_mr_aligned)
    amp_hyb_full = np.append(amp_insp[:t1_index_insp], amp_hyb_window, amp_mr_aligned[t2_index_mr_tent - 1:])
    # frq_hyb_full = np.append(frq_insp[:t1_index_insp], frq_hyb_window, frq_mr_aligned[t2_index_mr_tent -1:])


    waveform_hyb =  amp_hyb_full*np.exp(-1j*phase_hyb_full)

    return waveform_hyb, t1_index_insp, t1_index_mr_tent, t2_index_insp, t2_index_mr_tent, frq_insp, frq_mr, frq_mr_aligned, merger_ringdown_aligned, sample_indices_insp, sample_indices_mr, amp_hyb_window, amp_hyb_full, phase_hyb_window, phase_hyb_full



