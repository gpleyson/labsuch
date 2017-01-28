from __future__ import print_function

import os
import numpy as np
import pandas as pd

from scipy.interpolate import InterpolatedUnivariateSpline
from itertools import izip


class LabuschParameters(object):
    """
    Class containing Labusch parameters.
    """
    def __init__(self, name='unnamed', host='unkown', solute='unknown',
                 nty0=np.nan, nDEb=np.nan, nzeta=np.nan, wc=np.nan,
                 Gamma=np.nan):
        self.name = name            # name associated with parameters
        self.host = host            # name of host metal
        self.solute = solute        # name of solute
        self.nty0 = nty0            # zero-temperature yield stress (MPa)
        self.nDEb = nDEb            # characteristic energy barrier (eV)
        self.nzeta = nzeta          # characteristic length (A)
        self._Gamma = Gamma         # dislocation line tension (eV/A)
        self.wc = wc                # characteristic amplitude (A)

    @property
    def Gamma(self):
        """ Returns the dislocation line tension. """
        return self._Gamma

    @Gamma.setter
    def Gamma(self, new_Gamma):
        """
        Set new line tension Gamma and adjust the Labusch
        parameters accordingly.
        """
        self.nDEb = self.nDEb*(new_Gamma/self._Gamma)**(1./3.)
        self.nty0 = self.nty0/(new_Gamma/self._Gamma)**(1./3.)
        self.nzerta = self.nzeta*(new_Gamma/self._Gamma)**(2./3.)
        self._Gamma = new_Gamma

    def print_parameters(self):
        """
        Prints paramters to screen.
        """
        print('name  : {}'.format(self.name))
        print('host  : {}'.format(self.host))
        print('solute: {}'.format(self.solute))
        print('nty0  : {:6.4f} MPa'.format(self.nty0))
        print('nDEb  : {:6.4f} eV'.format(self.nDEb))
        print('nzeta : {:6.4f} A'.format(self.nzeta))
        print('wc    : {:6.4f} A'.format(self.wc))
        print('Gamma : {:6.4f} eV/A'.format(self._Gamma))

    def save(self, filename,  directory='.'):
        """
        Saves data to disk.

        :params:
            filename, directory
        """
        np.savez_compressed(os.path.join(directory, filename),
                            name=self.name,
                            host=self.host,
                            solute=self.solute,
                            nty0=self.nty0,
                            nDEb=self.nDEb,
                            nzeta=self.nzeta,
                            Gamma=self._Gamma,
                            wc=self.wc)
        return

    def load(self, filename,  directory='.'):
        """
        Loads data from disk.

        :params:
            filename, directory
        """
        data = np.load(os.path.join(directory, filename))
        self.name = str(data['name'])
        self.host = str(data['host'])
        self.solute = str(data['solute'])
        self.nty0 = float(data['nty0'])
        self.nDEb = float(data['nDEb'])
        self.nzeta = float(data['nzeta'])
        self._Gamma = float(data['Gamma'])
        self.wc = float(data['wc'])
        data.close()

        return


class SoluteStrengtheningModel(object):
    """
    Class containing for predicting finite temperature yield strength.
    """
    def __init__(self, parameters, conc, ep=1.e-4, ep0=1.e5, Gamma=np.nan,
                 ty_ath=0.):
        self._kb = 8.6173324e-5             # eV/K
        self.parameters = parameters        # Labusch parmeters
        self.conc = conc                    # list of concentrations (unitless)
        self.ep = ep                        # experimental strain rate
        self.ep0 = ep0                      # reference strain rate
        self.ty_ath = ty_ath                # athermal contribution (MPa)
        self.Cl = 0.55                      # constant for the high T soln

        if np.isnan(Gamma):
            self._Gamma = parameters[0].Gamma
        else:
            self._Gamma = Gamma

        self._get_effective_parameters()
        self.fit_strength_full()

    def _normalize_Gamma(self):
        """ Normalizes all parameters to have the same line tension value. """
        for ii in xrange(len(self.parameters)):
            self.parameters[ii].Gamma = self._Gamma

    @property
    def Gamma(self):
        """ Returns the dislocation line tension. """
        return self._Gamma

    @Gamma.setter
    def Gamma(self, new_Gamma):
        """
        Set new line tension Gamma and adjust the Labusch
        parameters accordingly.
        """
        self._Gamma = new_Gamma
        self._get_effective_parameters()
        self.fit_strength_full()

    def _get_effective_parameters(self):
        """
        Get effective zero-temperature yield strength of the alloy.
        """
        self._normalize_Gamma()

        tmp_stress = 0.
        tmp_barrier = 0.
        for ci, param in izip(self.conc, self.parameters):
            tmp_stress += ci*param.nty0**(3./2.)
            tmp_barrier += ci*param.nDEb**(3.)

        self.ty0 = tmp_stress**(2./3.)
        self.DEb = tmp_barrier**(1./3.)

    def predict_strength_LT(self, Tarray):
        """
        Predict finte temperature yield strength using the low
        temperature solution.
        """
        Tref = self.DEb / (self._kb * np.log(self.ep0/self.ep))

        tau_y = np.zeros(len(Tarray))

        mask = ((np.array(Tarray)/Tref) < 1.)
        tau_y[mask] = self.ty0*(1.-(Tarray[mask]/Tref)**(2./3.))

        return tau_y + self.ty_ath

    def predict_strength_HT(self, Tarray):
        """
        Predict finte temperature yield strength using the high
        temperature solution.
        """
        Tref = self.DEb / (self._kb * np.log(self.ep0/self.ep))
        tau_y = self.ty0*np.exp(-(1./self.Cl)*np.array(Tarray)/Tref)

        return tau_y + self.ty_ath

    def predict_strength(self, Tarray):
        """
        Predict finte temperature yield strength using the interpolated
        solution.
        """
        return self.ty_fit(Tarray) + self.ty_ath

    def fit_strength_full(self, taunorm_lim_lowT=0.6, taunorm_lim_highT=0.4,
                          Tmax=2000., Tstep=1.):
        """
        Fits a univariate spline between the low and high temperature
        solutions.

        :params:
            taunorm_lim_lowT: use up to this value for (ty/ty0) using the low
                    temperature solution
            taunorm_lim_highT: start with this value for (ty/ty0) for the
                    high temperature solution
            Tmax: fit up to this temperature
            Tstep: step in temperature
        """
        Tref = self.DEb / (self._kb * np.log(self.ep0/self.ep))

        Tmin_lowT = 0.
        Tmax_lowT = Tref*(1-taunorm_lim_lowT)**(3./2.)
        Tmin_highT = -Tref*self.Cl*np.log(taunorm_lim_highT)
        Tmax_highT = Tmax

        T_lowT = np.arange(Tmin_lowT, Tmax_lowT+Tstep, Tstep)
        T_highT = np.arange(Tmin_highT, Tmax_highT+Tstep, Tstep)
        T_full = np.append(T_lowT, T_highT)

        ty = np.append(self.predict_strength_LT(T_lowT)-self.ty_ath,
                       self.predict_strength_HT(T_highT)-self.ty_ath)
        self.ty_fit = InterpolatedUnivariateSpline(T_full, ty)

        return

    def print_system(self):
        """
        Print system information.
        """
        df_experiment = pd.DataFrame([[self.ty0, self.DEb, self.Gamma,
                                      self.ep, self.ep0]],
                                     columns=['t_y0 (MPa)', 'DE_b (eV)',
                                              'Gamma (eV/A)', 'epsilon',
                                              'epsilon_0'],
                                     index=['parameter'])

        df_solutes = pd.DataFrame()
        for params, conc in izip(self.parameters, self.conc):
            ser_tmp = pd.Series([params.host, params.solute, conc,
                                 params.nty0, params.nDEb],
                                index=['host', 'solute', 'conc',
                                       'nt_y0 (MPa)', 'nDE_b (eV)'],
                                name=params.name)
            df_solutes = pd.concat([df_solutes, ser_tmp.to_frame()], axis=1)

        print('General parameters')
        print('------------------')
        print(df_experiment)
        print()
        print('Solute parameters')
        print('-----------------')
        print(df_solutes)


class SoluteStrengtheningMinization(object):
    """
    Class to calculate normalized parameters from solute-dislocation
    interaction energies using the modified Labsuch model.

    :params:
        Uint(numpy ndarray): solute-dislocaiton interaction energy
        wstep(float): the incremental distance traveled by the
                      dislocation in (A)
        bvec(float) : the Burgers vector in (A)
        Gamma(float): the dislocation line tension in (eV/A)
        ishift_max(int): max increments used in the minimization
        name(string): name of the system
        host(string): name of host matrix
        solute(string): name of solute
    """
    def __init__(self, Uint=np.nan, wstep=np.nan, bvec=np.nan, Gamma=np.nan,
                 ishift_max=100, name='Unnamed', host='unkown',
                 solute='unkown'):
        self.Uint = Uint
        self.wstep = wstep
        self.bvec = bvec
        self._Gamma = Gamma
        self.ishift_max = ishift_max
        self.name = name
        self.host = host
        self.solute = solute

        self._warray = np.arange(ishift_max)*wstep
        self._calc_DEphat()
        self._calc_idmin()

    def _calc_DEphat(self):
        """
        """
        Uint = self.Uint
        self._DEphat = np.zeros(self.ishift_max)

        for ii in xrange(self.ishift_max):
            self._DEphat[ii] = (((Uint-np.roll(Uint, ii,
                                               axis=1))**2.).sum())**(1./2.)

        return

    def _calc_idmin(self):
        """
        Calculate total energy per unit length as a function of roughening
        amplitude w.
        """
        self._DEtotpL = -(3.**(2./3.)/(8.*2.**(1./3.)) *
                          (self._DEphat**4./(self.bvec**2.
                                             *(self._warray + 1.e-50)**2. 
                                             *self.Gamma))**(1./3.))
        self._DEtotpL[0] = 0.

        self._idmin = np.nonzero(self._DEtotpL == self._DEtotpL.min())[0][0]

    def get_Labusch_params(self):
        """Returns the Labusch paramters"""
        return LabuschParameters(name=self.name, host=self.host,
                                 solute=self.solute, nty0=self.nty0,
                                 nDEb=self.nDEb, nzeta=self.nzeta,
                                 Gamma=self.Gamma, wc=self.wc)

    @property
    def Gamma(self):
        """The dislocation line tension in (eV/A)"""
        return self._Gamma

    @Gamma.setter
    def Gamma(self, new_Gamma):
        """
        Set new line tension Gamma and adjust the Labusch
        parameters accordingly.
        """
        self._Gamma = new_Gamma
        self._calc_DEphat()
        # self._calc_idmin()

    @property
    def wc(self):
        """The characteristic roughening amplitude in (A)"""
        return self._warray[self._idmin]

    @property
    def nzeta(self):
        """The normalized characteristic dislocation length in (A)"""
        nzeta = (4.*3.**(1./2.) *
                 (self._Gamma**2. * self.wc**4. * self.bvec /
                  self._DEphat[self._idmin]**2.))**(1./3.)
        return nzeta

    @property
    def nDEc(self):
        """
        *Average* energy change associated by the bowed out configuration
        with characteristic length scales zeta_c and w_c
        """
        nDEc = -(3.**(5./6.)/2.**(5./3.) *
                 (self.wc**2. * self._Gamma *
                  (self._DEphat[self._idmin])**2./self.bvec)**(1./3.))
        return nDEc

    @property
    def nDEb(self):
        """The normalized characteristic energy barrier in (eV)"""
        return -(4.*np.sqrt(2.) - 1.)/3. * self.nDEc

    @property
    def nty0(self):
        """The normalized zero-temperature yield stress in (MPa)"""
        eVpA3_to_MPa = 1.6e-19*1e30*1e-6
        return (np.pi/2.*self.nDEb/(self.bvec*self.nzeta*self.wc) *
                eVpA3_to_MPa)

    def print_parameters(self):
        """Prints paramters to screen"""
        print('nty0  : {:6.4f} MPa'.format(self.nty0))
        print('nDEb  : {:6.4f} eV'.format(self.nDEb))
        print('nzeta : {:6.4f} A'.format(self.nzeta))
        print('wc    : {:6.4f} A'.format(self.wc))
        print('Gamma : {:6.4f} eV/A'.format(self.Gamma))


if __name__ == '__main__':
    name = 'test'
    host = 'host'
    solute = 'solute'
    nDEb = np.pi
    nty0 = np.pi
    wc = np.pi
    nzeta = np.pi
    Gamma = np.pi

    test = LabuschParameters(name=name, host=host, solute=solute,
                             nty0=nty0, nDEb=nDEb, nzeta=nzeta,
                             wc=wc, Gamma=Gamma)
    test.print_parameters()
