#!/bin/env python





if True:
    def corner_plot(self, params_plot,
                    params_true_vals=None, # TRUE / KNOWN values of PARAMETERS
                    params_oned_priors=None, # PRIOR DISTRIBUTION FOR PARAMETERS
                    fig=None, # CAN PLOT ON EXISTING FIGURE
                    axes_array=None, # NEED ARRAY OF AXES IF PLOTTING ON EXISTING FIGURE
                    histogram_type='bar', # Histogram type (bar / step / barstacked)
                    nhbins=30,  # NO OF BINS IN HISTOGRAMS
                    projection='rectilinear',
                    label='', # LABEL THAT GOES ON EACH PANEL
                    params_labels=None,
                    plim_low=None, plim_high=None,
                    legend_fontsize=18,
                    plot_type='scatter', # SCATTER OR CONTOUR
                    color=None,
                    hist_alpha=0.3, # Transparency of histograms
                    scatter_alpha=0.2, # Transparency of scatter points
                    npixels=50,
                    param_color=None, # 3RD DIMENSION SHOWN AS COLOR
                    param_color_label=None, # LABEL of 3RD DIMENSION SHOWN AS COLOR
                    color_max=None,
                    color_min=None,
                    cmap=cm.plasma_r,
                    contour_levels=[90.0],
                    contour_lstyles=["solid" , "dashed" , "dashdot" , "dotted"],
                    label_contours=True, #Whether or not to label individuals
                    contour_labels_inline=True,
                    contour_labels_loc="upper center",
                    return_areas_in_contours=False,
                    label_oned_hists=-1, # Which one-d histograms to label?
                    skip_oned_hists=False,
                    label_oned_loc='outside',
                    show_oned_median=False,
                    grid_oned_on=False,
                    debug=False, verbose=None
                    ):
        """
Generates a corner plot for given parameters. 2D panels can have data points
directly or percentile contours, not both simultaneously yet.

When plotting data points, user can also add colors to it based on a 3rd parameter
[Use param_color].

 When plotting contours, user can ask for the area inside the contours to be
 returned. However, if the contours are railing against boundaries, there is no
 guarantee that the areas will be correct. Multiple disjoint closed contours are
 supported though [Use return_areas_in_contours].

Input:
(1) [REQUIRED] params_plot: list of parameters to plot. String names or Indices work.
(2) [OPTIONAL] params_labels: list of parameter names to use in latex labels.
(2a,b) [OPTIONAL] xlim_low/high: lists of parameter lower & upper limits
(3) [REQUIRED] plot_type: Either "scatter" or "contour"
(4) [OPTIONAL] contour_levels: credible interval levels for "contour" plots
(5) [OPTIONAL] contour_lstyles: line styles for contours. REQUIRED if >4 contour levels
                                are to be plotted.
(6) [OPTIONAL] npixels: number of pixels/bins along each dimension in contour plots.
(7) [OPTIONAL] label: String label to label the entire data.
(8) [OPTIONAL] color / param_color: Single color for "scatter"/"contour" plots
                                          OR
                        One parameter that 2D scatter plots will show as color.
(9) [OPTIONAL] color_max=None, color_min=None: MAX/MIN values of parameter
                                              "param_color" to use in "scatter"
                                              plots
(10)[OPTIONAL] nhbins=30 : NO OF BINS IN HISTOGRAMS
(11)[OPTIONAL] params_oned_priors=None: PRIOR SAMPLES to be overplotted onto
                                        1D histograms. Dictionary.
        """
        ## Preliminary checks on inputs
        if len(contour_levels) > len(contour_lstyles):
            if plot_type == 'contour':
                raise IOError("Please provide as many linestyles as contour levels")

        if param_color is not None and "scatter" not in plot_type:
            raise IOError("Since you passed a 3rd dimension, only plot_type=scatter is allowed")

        ## Local verbosity level takes precedence, else the class's is used
        if verbose == None: verbose = self.verbose

        ## IF no labels are provided by User, use default Latex labels for CBCs
        if params_labels is None:
            params_labels = ParamLatexLabels()

        def get_param_label(pp):
            if params_labels is not None and pp in params_labels:
                return params_labels[pp]
            if pp in param_color and param_color_label is not None:
                return param_color_label
            return pp.replace('_','-')

        ## This is the label for the entire data set, not individual quantities
        label      = label.replace('_','-')

        ## Configure the full figure panel
        no_of_rows = len(params_plot)
        no_of_cols = len(params_plot)

        if type(fig) != matplotlib.figure.Figure or axes_array is None:
            #fig = plt.figure(figsize=(6*no_of_cols,4*no_of_rows))
            fig, axes_array = plt.subplots(no_of_rows, no_of_cols,
                figsize=(6*no_of_cols,4*no_of_rows),
                gridspec_kw = {'wspace':0, 'hspace':0})

        fig.hold(True)

        ## Pre-choose color for 1D histograms (and scatter plots, if applicable)
        rand_color = np.random.rand(3)
        if color != None: rand_color = color

        if return_areas_in_contours: contour_areas = {}
        contour_levels = sorted(contour_levels, reverse=True)

        ## Start drawing panels
        for nr in range(no_of_rows):
            for nc in range(no_of_cols):
                ## We keep the upper diagonal half of the figure empty.
                ## FIXME: Could we use it for same data, different visualization?
                if nc > nr:
                    ax = axes_array[nr][nc]
                    try:
                        fig.delaxes(ax)
                    except: pass
                    continue

                # Make 1D histograms along the diagonal
                if nc == nr:
                    if skip_oned_hists: continue
                    p1 = params_plot[nc]
                    p1label = get_param_label(p1)
                    #ax = fig.add_subplot(no_of_rows, no_of_cols, (nr*no_of_cols) + nc + 1)
                    if no_of_rows == 1 and no_of_cols == 1:
                        ax = axes_array
                    else:
                        ax = axes_array[nr][nc]
                    # Plot known / injected / true value if given
                    if params_true_vals != None:
                        p_true_val = params_true_vals[nc]
                        if p_true_val != None:
                            ax.axvline(p_true_val,
                                       lw = 0.5, ls='solid', color = rand_color)
                    # Plot one-d posterior
                    _data = self.sliced(p1).data()
                    im = ax.hist(_data, bins=nhbins,
                                  histtype=histogram_type,
                                  normed=True, alpha=hist_alpha,
                                  color=rand_color, label=label)
                    ax.axvline(np.percentile(_data, 5), lw=1, ls = 'dashed', color = rand_color, alpha=1)
                    ax.axvline(np.percentile(_data, 95), lw=1, ls = 'dashed', color = rand_color, alpha=1)
                    if show_oned_median:
                        ax.axvline(np.median(_data), ls = '-', color = rand_color)
                    try:
                        if label_oned_hists == -1 or nc in label_oned_hists:
                            if label_oned_loc is not 'outside' and label_oned_loc is not '':
                                ax.legend(loc=label_oned_loc, fontsize=legend_fontsize)
                            else:
                                ax.legend(fontsize=legend_fontsize, bbox_to_anchor=(1.3, 0.9))
                    except TypeError: raise TypeError("Pass a list for label_oned_hists")
                    if params_oned_priors is not None and p1 in params_oned_priors:
                        _data = params_oned_priors[p1]
                        if plim_low is not None and plim_high is not None:
                            _prior_xrange = (plim_low[nc], plim_high[nc])
                        else: _prior_xrange = None
                        im = ax.hist(_data, bins=nhbins,
                                    histtype="step", color='k',
                                    range=_prior_xrange, normed=True
                                    )
                    if plim_low is not None and plim_high is not None:
                        ax.set_xlim(plim_low[nc], plim_high[nc])
                    ax.grid(grid_oned_on)
                    if nr == (no_of_rows-1): ax.set_xlabel(p1label)
                    if nc == 0 and (no_of_cols > 1 or no_of_rows > 1):
                        ax.set_ylabel(p1label)
                    if nr < (no_of_rows-1):
                        ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    continue

                ## If execution reaches here, the current panel is in the lower diagonal half
                if verbose:
                    print "Making plot (%d,%d,%d)" % (no_of_rows, no_of_cols, (nr*no_of_cols) + nc)

                ## Get plot for this panel
                #ax = fig.add_subplot(no_of_rows, no_of_cols, (nr*no_of_cols) + nc + 1,
                #                          projection=projection)
                ax = axes_array[nr][nc]

                ## Plot known / injected / true value if given
                if params_true_vals != None:
                    pc_true_val = params_true_vals[nc]
                    pr_true_val = params_true_vals[nr]
                    if pc_true_val != None:
                        ax.axvline(pc_true_val,
                                   lw = 0.5, ls='solid', color = rand_color)
                    if pr_true_val != None:
                        ax.axhline(pr_true_val,
                                   lw = 0.5, ls='solid', color = rand_color)
                    if pc_true_val != None and pr_true_val != None:
                        ax.plot([pc_true_val], [pr_true_val],
                                's', color = rand_color)

                ## Now plot what the user requested
                ### If user asks for scatter-point colors to be a 3rd dimension
                if param_color in self.var_names:
                    p1 = params_plot[nc]
                    p2 = params_plot[nr]
                    p1label = get_param_label(p1)
                    p2label = get_param_label(p2)
                    cblabel = get_param_label(param_color)
                    if verbose:
                        print "Scatter plot w color: %s vs %s vs %s" % (p1,p2, param_color)
                    _d1, _d2  = self.sliced(p1).data(), self.sliced(p2).data()
                    im = ax.scatter(_d1, _d2, c=self.sliced(param_color).data(),
                                    alpha=scatter_alpha,
                                    edgecolors=None, linewidths=0,
                                    vmin=color_min, vmax=color_max, cmap=cmap,
                                    label=label)
                    cb = fig.colorbar(im, ax=ax)
                    if nc == (no_of_cols-1): cb.set_label(cblabel)
                    if nr == (no_of_rows-1): ax.set_xlabel(p1label)
                    if nc == 0: ax.set_ylabel(p2label)
                    ## set X/Y axis limits
                    if plim_low is not None and plim_high is not None:
                        ax.set_xlim( plim_low[nc], plim_high[nc] )
                        ax.set_ylim( plim_low[nr], plim_high[nr] )
                    else:
                        ax.set_xlim( 0.95 * np.min(_d1), 1.05 * np.max(_d1) )
                        ax.set_ylim( 0.95 * np.min(_d2), 1.05 * np.max(_d2) )
                    ax.legend(loc='best', fontsize=legend_fontsize)
                    ax.grid()
                elif param_color is not None:
                    raise IOError("Could not find parameter %s to show" % param_color)
                ### If user asks for scatter plot without 3rd Dimension info
                elif plot_type=='scatter':
                    p1 = params_plot[nc]
                    p2 = params_plot[nr]
                    p1label = get_param_label(p1)
                    p2label = get_param_label(p2)
                    if verbose: print "Scatter plot: %s vs %s" % (p1,p2)
                    _d1, _d2  = self.sliced(p1).data(), self.sliced(p2).data()
                    im = ax.scatter(_d1, _d2, c=rand_color,
                                    alpha=scatter_alpha,
                                    edgecolors=None, linewidths=0,
                                    label=label)
                    if nr == (no_of_rows-1): ax.set_xlabel(p1label)
                    if nc == 0: ax.set_ylabel(p2label)
                    if plim_low is not None and plim_high is not None:
                        ax.set_xlim( plim_low[nc], plim_high[nc] )
                        ax.set_ylim( plim_low[nr], plim_high[nr] )
                    else:
                        ax.set_xlim( 0.95 * np.min(_d1), 1.05 * np.max(_d1) )
                        ax.set_ylim( 0.95 * np.min(_d2), 1.05 * np.max(_d2) )
                    ax.legend(loc='best', fontsize=legend_fontsize)
                    ax.grid()
                ### If user asks for contour plot without 3rd Dimension info
                elif plot_type=='contour':
                    p1 = params_plot[nc]
                    p2 = params_plot[nr]
                    p1label = get_param_label(p1)
                    p2label = get_param_label(p2)
                    if verbose: print "Contour plot: %s vs %s" % (p1,p2)
                    ## Get data
                    d1 = self.sliced(p1).data()
                    d2 = self.sliced(p2).data()
                    dd = np.column_stack([d1, d2])
                    if verbose: print np.shape(d1), np.shape(d2), np.shape(dd)
                    pdf = gaussian_kde(dd.T)
                    ## Get contour levels
                    zlevels = [np.percentile(pdf(dd.T), 100.0 - lev) for lev in contour_levels]
                    x11vals = np.linspace(dd[:,0].min(), dd[:,0].max(),npixels)
                    x12vals = np.linspace(dd[:,1].min(), dd[:,1].max(),npixels)
                    q, w = np.meshgrid(x11vals, x12vals)
                    r1 = pdf([q.flatten(),w.flatten()])
                    r1.shape=q.shape
                    ## Draw contours
                    im = ax.contour(x11vals, x12vals, r1, zlevels,
                                    colors=rand_color,
                                    linestyles=contour_lstyles[:len(contour_levels)],
                                    label=label)

                    ## Get area inside contour
                    if return_areas_in_contours:
                        if verbose: print "Computing area inside contours."
                        contour_areas[p1+p2] = []
                        for ii in range(len(zlevels)):
                            contour = im.collections[ii]
                            # Add areas inside all independent contours, in case
                            # there are multiple disconnected ones
                            contour_areas[p1+p2].append(\
                                  np.sum([area_inside_contour(vs.vertices)\
                                    for vs in contour.get_paths()]) )
                            if verbose:
                                print "Total area = %.9f, %.9f" % (contour_areas[p1+p2][-1])
                            if debug:
                                for _i, vs in enumerate(contour.get_paths()):
                                    print "sub-area %d: %.8e" % (_i, area_inside_contour(vs.vertices))
                        contour_areas[p1+p2] = np.array( contour_areas[p1+p2] )

                    ####
                    ## BEAUTIFY contour labeling..!
                    # Define a class that forces representation of float to look
                    # a certain way. This remove trailing zero so '1.0' becomes '1'
                    class nf(float):
                        def __repr__(self):
                            str = '%.1f' % (self.__float__(),)
                            if str[-1] == '0': return '%.0f' % self.__float__()
                            else: return '%.1f' % self.__float__()
                    # Recast levels to new class
                    im.levels = [nf(val) for val in contour_levels]
                    # Label levels with specially formatted floats
                    if plt.rcParams["text.usetex"]: fmt = r'%r \%%'
                    else: fmt = '%r %%'
                    ####
                    if label_contours:
                        if contour_labels_inline:
                            ax.clabel(im, im.levels,
                                  inline=False,
                                  use_clabeltext=True,
                                  fmt=fmt, fontsize=10)
                        else:
                            for zdx, _ in enumerate(zlevels):
                                _ = ax.plot([], [], color=rand_color,
                                            ls=contour_lstyles[:len(contour_levels)][zdx],
                                            label=im.levels[zdx])
                            ax.legend(loc=contour_labels_loc, fontsize=legend_fontsize)
                    else: pass
                    #
                    if nr == (no_of_rows-1): ax.set_xlabel(p1label)
                    if nc == 0: ax.set_ylabel(p2label)
                    if plim_low is not None and plim_high is not None:
                        ax.set_xlim( plim_low[nc], plim_high[nc] )
                        ax.set_ylim( plim_low[nr], plim_high[nr] )
                    elif projection != "mollweide":
                        ax.set_xlim( 0.95 * np.min(d1), 1.05 * np.max(d1) )
                        ax.set_ylim( 0.95 * np.min(d2), 1.05 * np.max(d2) )
                    else: pass
                    #ax.legend(loc='best', fontsize=legend_fontsize)
                    ax.grid(True)
                else:
                    raise IOError("plot type %s not supported.." % plot_type)
                if nc != 0:
                    print "removing Yticklabels for (%d, %d)" % (nr, nc)
                    ax.set_yticklabels([])
                if nr != (no_of_rows - 1):
                    print "removing Xticklabels for (%d, %d)" % (nr, nc)
                    ax.set_xticklabels([])
        ##
        for nc in range(1, no_of_cols):
            ax = axes_array[no_of_rows - 1][nc]
            #new_xticklabels = [ll.get_text() for ll in ax.get_xticklabels()]
            new_xticklabels = ax.get_xticks().tolist()
            new_xticklabels[0] = ''
            ax.set_xticklabels(new_xticklabels)
        #fig.subplots_adjust(wspace=0, hspace=0)
        if plot_type=='contour' and return_areas_in_contours and debug:
            return fig, axes_array, contour_areas, contour.get_paths(), im
        elif plot_type=='contour' and return_areas_in_contours:
            return fig, axes_array, contour_areas
        else:
            return fig, axes_array
    ##

