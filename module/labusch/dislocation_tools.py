import numpy as np
from collections import namedtuple
from itertools import izip

import matplotlib.pyplot as plt


Coordinates_2d = namedtuple('Coordinates_2d', ['x', 'y'])

ElasticConstants = namedtuple('Elastic_Constants', ['mu', 'nu'])

DisplacementField = namedtuple('Displacement_Field', ['ux', 'uy', 'uz'])

StressField = namedtuple('Stress_Field', ['sxx', 'syy', 'szz',
                                          'syz', 'sxz', 'sxy'])

StrainField = namedtuple('Strain_Field', ['exx', 'eyy', 'ezz',
                                          'eyz', 'exz', 'exy'])


class DislocationArray(object):
    """
    Displacement and stress field of an array of dislocations.

    :attributes:
        elconst(ElasticConstants): elastic constants of the system (mu in GPa)
        rundef(Coordinates_2d): undeformed coordinates of atoms
        bedge_array(array): array of the Burgers vector magnitude of
                            the edge components
        bscrew_array(array): array of the Burgers vector magnitude of
                             the screw components
        xdisl_array(array): array of x-coordinate of dislocations
        ydisl_array(array): array of y-coordinate of dislocations
        rdef(Coordinates_2d): deformed coordinates of atoms

        bvec(float): effective Burgers vector magnitude of dislocations
        bedge(float): effective edge Burgers vecter magnitude of dislocations
        bscrew(float): effective screw Burgers vecter magnitude of dislocations
        udisl(DisplacementField): displacement field of dislocations
        sdisl(StressField): stress field of dislocations
    """
    def __init__(self, elconst, rundef, bedge_array, bscrew_array,
                 xdisl_array, ydisl_array):
        self.elconst = elconst
        self.rundef = rundef
        self.bedge_array = bedge_array
        self.bscrew_array = bscrew_array
        self.xdisl_array = xdisl_array
        self.ydisl_array = ydisl_array

        self._calc_fields()

    @property
    def bvec(self):
        """Returns the effective Burgers vector of the system. """
        return np.sqrt(np.sum(self.bedge_array)**2. +
                       np.sum(self.bscrew_array)**2.)

    @property
    def bedge(self):
        """Returns the effective edge Burgers vector of the system. """
        return np.sum(self.bedge_array)

    @property
    def bscrew(self):
        """Returns the effective screw Burgers vector of the system. """
        return np.sum(self.bscrew_array)

    def _calc_fields(self):
        """
        Calculates the total displacement and stress fields of the system.
        """
        # initialize fields
        ux = np.zeros(self.rundef.x.shape)
        uy = np.zeros(self.rundef.x.shape)
        uz = np.zeros(self.rundef.x.shape)
        sxx = np.zeros(self.rundef.x.shape)
        syy = np.zeros(self.rundef.x.shape)
        szz = np.zeros(self.rundef.x.shape)
        syz = np.zeros(self.rundef.x.shape)
        sxz = np.zeros(self.rundef.x.shape)
        sxy = np.zeros(self.rundef.x.shape)

        # loop over all dislocations
        for bedge, bscrew, xdisl, ydisl in izip(self.bedge_array,
                                                self.bscrew_array,
                                                self.xdisl_array,
                                                self.ydisl_array):
            # set dislocation position
            rdisl = Coordinates_2d(x=xdisl, y=ydisl)

            # calculate displacements
            uedge = displacement_edge_Volterra_ILE(rdisl, self.rundef,
                                                   self.elconst, bedge)
            uscrew = displacement_screw_Volterra_ILE(rdisl, self.rundef,
                                                     self.elconst, bscrew)

            # caclulate stress fields
            sedge = stress_edge_Volterra_ILE(rdisl, self.rundef,
                                             self.elconst, bedge)
            sscrew = stress_screw_Volterra_ILE(rdisl, self.rundef,
                                               self.elconst, bscrew)

            # update cumulative variables
            ux += uedge.ux + uscrew.ux
            uy += uedge.uy + uscrew.uy
            uz += uedge.uz + uscrew.uz
            sxx += sedge.sxx + sscrew.sxx
            syy += sedge.syy + sscrew.syy
            szz += sedge.szz + sscrew.szz
            syz += sedge.syz + sscrew.syz
            sxz += sedge.sxz + sscrew.sxz
            sxy += sedge.sxy + sscrew.sxy

        # assign fields
        self.udisl = DisplacementField(ux=ux, uy=uy, uz=uz)
        self.sdisl = StressField(sxx=sxx, syy=syy, szz=szz,
                                 syz=syz, sxz=sxz, sxy=sxy)
        self.rdef = Coordinates_2d(x=self.rundef.x+ux, y=self.rundef.y+uy)

    def calc_interaction_energy(self, dv):
        """
        Calculate interaction energy of the dislocations with a point
        dilatational source with misfit volume dv.

        :params:
            dv(float): misfit volume of point dilatational source

        :returns:
            Uint(nd-array): interaction energy in eV
        """
        GPaA3_to_eV = 1e9*1e-30/1.6e-19

        pressure = (self.sdisl.sxx+self.sdisl.syy+self.sdisl.szz)/3.

        return -pressure*dv * GPaA3_to_eV


class SpreadCoreDislocation(DislocationArray):
    """
    Implementation of the spread core dislcoation.

    :attributes:
        elconst(ElasticConstants): elastic constants of the system (mu in GPa)
        rundef(Coordinates_2d): undeformed coordinates of atoms
        bedge(float): effective edge Burgers vecter magnitude of dislocations
        bscrew(float): effective screw Burgers vecter magnitude of dislocations
        rdisl_center(Coordinates_2d): coordinate of disloc. center
        disl_spacing(float): spacing of dislocation array
        ndisl(int): number of dislocation

        bedge_array(array): array of the Burgers vector magnitude of
                            the edge components
        bscrew_array(array): array of the Burgers vector magnitude of
                             the screw components
        xdisl_array(array): array of x-coordinate of dislocations
        ydisl_array(array): array of y-coordinate of dislocations
        rdef(Coordinates_2d): deformed coordinates of atoms

        bvec(float): effective Burgers vector magnitude of dislocations
        udisl(DisplacementField): displacement field of dislocations
        sdisl(StressField): stress field of dislocations
    """

    def __init__(self, elconst, rundef, bedge, bscrew, disl_spacing,
                 ndisl=11, rdisl_center=Coordinates_2d(x=0., y=0.)):
        self.xdisl_array = (np.arange(ndisl) -
                            (rdisl_center.x+(ndisl-1)/2.))*disl_spacing
        self.ydisl_array = np.ones(ndisl) * rdisl_center.y
        self.bedge_array = bedge * np.ones(ndisl)/float(ndisl)
        self.bscrew_array = bscrew * np.ones(ndisl)/float(ndisl)

        self.elconst = elconst
        self.rundef = rundef

        self._calc_fields()


def displacement_edge_Volterra_ILE(rdisl, ratoms, elconst, bedge):
    """
    Returns the displacement field of a Volterra edge dislocation
    on the crystal using isotropic linear elasticity.

    :params:
        rdisl(Coordinates_2d): coordinates of the dislocation
        ratoms(Coordinates_2d): coordinates of the atoms
        elconst(ElasticConstants): elastic constants of the crystal
        bedge(float): magnitude of edge Burgers vector

    :returns:
        uedge(DisplacementField): displacement field of edge dislocation
    """
    # initialize dislacement field
    # shift coordinates so disloc. is at the origin
    rsh = Coordinates_2d(x=ratoms.x-rdisl.x,
                         y=ratoms.y-rdisl.y)

    # calculate displacements
    uxedge = bedge/2./np.pi*(np.arctan2(rsh.y, rsh.x) +
                             (rsh.x*rsh.y)/(2.*(1.-elconst.nu) *
                                            (rsh.x**2.+rsh.y**2.)))
    uyedge = -bedge/2./np.pi*((1.-elconst.nu)/(4.*(1.-elconst.nu)) *
                              np.log(rsh.x**2. + rsh.y**2.) +
                              (rsh.x**2.-rsh.y**2.)/(4.*(1.-elconst.nu) *
                                                     (rsh.x**2.+rsh.y**2.)))

    uedge = DisplacementField(ux=uxedge,
                              uy=uyedge,
                              uz=np.zeros(ratoms.x.shape))

    return uedge


def displacement_screw_Volterra_ILE(rdisl, ratoms, elconst, bscrew):
    """
    Returns the displacement field of a Volterra screw dislocation
    on the crystal using isotropic linear elasticity.

    :params:
        rdisl(Coordinates_2d): coordinates of the dislocation
        ratoms(Coordinates_2d): coordinates of the atoms
        elconst(ElasticConstants): elastic constants of the crystal
        bscrew(float): magnitude of screw Burgers vector

    :returns:
        uscrew(DisplacementField): displacement field of screw dislocation
    """
    # shift coordinates so disloc. is at the origin
    rsh = Coordinates_2d(x=ratoms.x-rdisl.x,
                         y=ratoms.y-rdisl.y)

    # calculate displacements
    uzscrew = bscrew/2./np.pi*np.arctan2(rsh.y, rsh.x)

    uscrew = DisplacementField(ux=np.zeros(ratoms.x.shape),
                               uy=np.zeros(ratoms.x.shape),
                               uz=uzscrew)

    return uscrew


def stress_edge_Volterra_ILE(rdisl, ratoms, elconst, bedge):
    """
    Returns the stress field of a Volterra edge dislocation
    on the crystal using isotropic linear elasticity.

    :params:
        rdisl(Coordinates_2d): coordinates of the dislocation
        ratoms(Coordinates_2d): coordinates of the atoms
        elconst(ElasticConstants): elastic constants of the crystal
        bscrew(float): magnitude of screw Burgers vector

    :returns:
        sigma(StressField): stress field of edge dislocation
    """
    # shift coordinates so disloc. is at the origin
    rsh = Coordinates_2d(x=ratoms.x-rdisl.x,
                         y=ratoms.y-rdisl.y)

    # calculate stress
    sxx = (-elconst.mu*bedge/(2.*np.pi*(1.-elconst.nu)) *
           (rsh.y*(3.*rsh.x**2.+rsh.y**2.))/(rsh.x**2.+rsh.y**2.)**2.)
    syy = (elconst.mu*bedge/(2.*np.pi*(1.-elconst.nu)) *
           (rsh.y*(rsh.x**2.-rsh.y**2.))/(rsh.x**2.+rsh.y**2.)**2.)
    szz = elconst.nu*(sxx+syy)
    sxy = (elconst.mu*bedge/(2.*np.pi*(1.-elconst.nu)) *
           (rsh.x*(rsh.x**2.-rsh.y**2.))/(rsh.x**2.+rsh.y**2.)**2.)
    sxz = np.zeros(rsh.x.shape)
    syz = np.zeros(rsh.x.shape)

    sigma = StressField(sxx=sxx, syy=syy, szz=szz, syz=syz, sxz=sxz, sxy=sxy)

    return sigma


def stress_screw_Volterra_ILE(rdisl, ratoms, elconst, bscrew):
    """
    Returns the stress field of a Volterra screw dislocation
    on the crystal using isotropic linear elasticity.

    :params:
        rdisl(Coordinates_2d): coordinates of the dislocation
        ratoms(Coordinates_2d): coordinates of the atoms
        elconst(ElasticConstants): elastic constants of the crystal
        bscrew(float): magnitude of screw Burgers vector

    :returns:
        sigma(StressField): stress field of screw dislocation
    """
    # shift coordinates so disloc. is at the origin
    rsh = Coordinates_2d(x=ratoms.x-rdisl.x,
                         y=ratoms.y-rdisl.y)

    # calculate stress
    sxz = elconst.mu*bscrew/(2.*np.pi)*(rsh.y/(rsh.x**2.+rsh.y**2.))
    syz = elconst.mu*bscrew/(2.*np.pi)*(rsh.x/(rsh.x**2.+rsh.y**2.))
    sxx = np.zeros(rsh.x.shape)
    syy = np.zeros(rsh.x.shape)
    szz = np.zeros(rsh.x.shape)
    sxy = np.zeros(rsh.x.shape)

    sigma = StressField(sxx=sxx, syy=syy, szz=szz, syz=syz, sxz=sxz, sxy=sxy)

    return sigma


if __name__ == '__main__':
    m = 15
    xvar = np.arange(-m, m+1)
    yvar = np.arange(-m, m) + 0.5
    xx, yy = np.meshgrid(xvar, yvar)
    ratoms = Coordinates_2d(x=xx, y=yy)

    elconst = ElasticConstants(mu=100., nu=0.3)
    ndisl = 11
    disl_spacing = .6

    bedge = 1.
    bscrew = 0.

    rdisl_center = Coordinates_2d(x=0., y=0.)
    disl = SpreadCoreDislocation(elconst, ratoms, bedge, bscrew, disl_spacing,
                                 ndisl=ndisl, rdisl_center=rdisl_center)

    x = disl.rdef.x
    y = disl.rdef.y
    c = (disl.sdisl.sxx+disl.sdisl.syy+disl.sdisl.szz)/3.
    dv = 1.
    c = disl.calc_interaction_energy(dv)

    plt.ion()
    plt.figure(1, figsize=(5, 5))
    plt.clf()
    plt.scatter(x, y, c=c, s=40, cmap='RdBu')
    plt.axis('equal')
    plt.colorbar()
