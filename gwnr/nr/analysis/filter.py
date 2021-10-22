# Copyright (C) 2019 Prayush Kumar
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
from __future__ import print_function

from gwnr.utils.support import *
from gwnr.waveform.condition import blend
import sys

from glue.ligolw import lsctables
from glue.ligolw import ligolw
import lal


@lsctables.use_in
class LIGOLWContentHandler(ligolw.LIGOLWContentHandler):
    pass


#############################
def overlaps_vs_totalmass(wav1,
                          wav2,
                          psd=None,
                          mf_lower=-1.,
                          m_lower=-1.,
                          m_upper=100.,
                          m_delta=5.):
    '''
Need two wobjects of nr_waveform class.

Waveforms are rescaled to different total masses and their overlaps
computed.

Returns an array of total masses and overlaps.
    '''
    # {{{
    # print(min(wav1.rawhp), max(wav1.rawhp), max(wav2.rawhp), min(wav2.rawhp))
    if psd is None:
        raise IOError("Provide the PSD please!")
    if mf_lower < 0:
        print("Initial orbital frequencies will be deduced after blending")
    #
    t2_opt = [1000, 2000]
    t_option = [100, t2_opt[0], t2_opt[1], 50, 100]
    f_lower = 15. - 0.5
    # Calculate lowest total mass, from a) mf_lower, b) m_lower, c) calculate
    if mf_lower > 0:
        m_lower = mf_lower / f_lower / lal.MTSUN_SI
    elif m_lower <= 0:
        rescaled_mass, orbit_freq1 = wav1.get_orbital_frequency(t=max(t2_opt))
        rescaled_mass, orbit_freq2 = wav2.get_orbital_frequency(t=max(t2_opt))
        m_lower = max(orbit_freq1, orbit_freq2) * rescaled_mass / f_lower
    print(orbit_freq1, orbit_freq2, "lowest total Mass = %f" % m_lower)
    #
    overlaps = []
    mass_range = get_uniform_mass_range(m_lower, m_upper, m_delta)
    for mtot in mass_range:
        # wav1.rescale_to_totalmass( mtot )
        # wav2.rescale_to_totalmass( mtot )
        wav_blended1 = blend(wav1, mtot, wav1.sample_rate, wav1.time_length,
                             t_option)  # blending
        wav_blended2 = blend(wav2, mtot, wav1.sample_rate, wav1.time_length,
                             t_option)  # blending
        if len(wav_blended1) != len(wav_blended2):
            raise RuntimeError(
                "blending function return different sets of waveforms!!")
        tmp_overlaps = [mtot]
        for ii in range(len(wav_blended1)):
            hp1, hp2 = wav_blended1[ii], wav_blended2[ii]
            olap = overlap_between_waveforms(hp1, hp2, psd=psd)
            tmp_overlaps.append(olap)
            print("--In OvsM: window %d, overlap = %f" % (ii, olap))
        overlaps.append(tmp_overlaps)
    return overlaps
    # }}}


def calculate_mismatch_between_levs_hdf5(
        self,
        wavefilename='rhOverM_CcePITT_Asymptotic_GeometricUnits.h5',
        outdir='matches',
        outputfile='OverlapsLevs.h5',
        catalogfile=None,
        m_upper=100.,
        m_delta=5.):
    # {{{
    cmd.getoutput('mkdir -p %s/%s' % (self.outdir, outdir))
    fout = h5py.File(self.outdir + '/' + outdir + '/' + outputfile, "a")
    #
    # Get the waveforms for different levs
    self.read_waveforms_from_hdf5_files(wavefilename=wavefilename)
    # Get PSD
    sample_rate, time_length = self.sample_rate, self.time_length
    N = sample_rate * time_length
    self.psd = self.get_psd()
    #
    ccefiles = list(self.wavefiles[self.levs[0]].keys())
    # ccefiles = list(np.sort( self.ccefiles[self.levs[0]] )[num_runs:])
    # Obtain the waveform files for given CceR, at Lev3,4,5
    # In pairs, compare Lev3,4,5
    self.levs.sort()
    for ccef in ccefiles:
        # choose a pair of levs
        for i1 in range(len(self.levs)):
            ld1 = self.levs[i1]
            for i2 in range(i1, len(self.levs)):  # Include self overlaps
                ld2 = self.levs[i2]
                if ccef not in list(self.hwaveforms[ld1].keys()) or \
                        ccef not in list(self.hwaveforms[ld2].keys()):
                    print(ccef, " waveforms not found in both %s and %s" %
                          (ld1, ld2))
                    continue
                # Create a group in output file for this ccefile
                if ccef not in list(fout.keys()):
                    fout.create_group(ccef)
                # Compute matches
                if self.verbose:
                    print("\n\nOverlaps for %s Between %s and %s" %
                          (ccef, ld1, ld2),
                          file=sys.stderr)
                overlaps = overlaps_vs_totalmass(self.hwaveforms[ld1][ccef],
                                                 self.hwaveforms[ld2][ccef],
                                                 psd=self.psd,
                                                 m_upper=m_upper,
                                                 m_delta=m_delta)
                # Add matches and masses as a dataset to the group
                dsetname = ld1 + '_' + ld2 + '.dat'
                fout[ccef].create_dataset(dsetname, data=overlaps)
    #
    fout.flush()
    fout.close()
    return
    # }}}
    #
