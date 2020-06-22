import geokit as gk
import pandas as pd
import numpy as np

from os import mkdir, environ
from os.path import join, isfile, isdir
from collections import OrderedDict, namedtuple
from types import FunctionType
from .. import core as rk_wind_core
from ...workflow_manager import WorkflowManager


class WindWorkflowManager(WorkflowManager):
    """
    Helps managing the logical workflow for simulations relating to wind turbines

    Initialization:

    Parameters
    ----------
    placements : pandas.DataFrame
        * A Pandas DataFrame describing the wind turbine placements to be simulated
        * Must include the following columns
            - 'geom' or 'lat' and 'lon'
            - 'hub_height'
            - 'capacity'
            - 'rotor_diam' or 'powerCurve' 

    synthetic_power_curve_cut_out 


    synthetic_power_curve_rounding 



    Returns
    -------
    numpy array
        A corrected power curve.

    """

    def __init__(self, placements, synthetic_power_curve_cut_out=25, synthetic_power_curve_rounding=1):
        #0
            powerCurve = []
            for sppow in specificPower:
                pcid = "SPC:%d,%d" % (sppow, synthetic_power_curve_cut_out)
                powerCurve.append(pcid)

            self.placements['powerCurve'] = powerCurve

        # Put power curves into the power curve library
        for pc in self.placements.powerCurve.values:
            assert isinstance(pc, str), \
                "Power curve value needs to be a string, not " + type(pc)

            if pc in self.powerCurveLibrary:
                continue

            if pc[:4] == "SPC:":
                sppow, cutout = pc.split(":")[1].split(",")
                self.powerCurveLibrary[pc] = rk_wind_core.power_curve.PowerCurve.from_specific_power(
                    specific_power=float(sppow),
                    cutout=float(cutout))
            else:
                self.powerCurveLibrary[pc] = rk_wind_core.turbine_library.TurbineLibrary().loc[pc].PowerCurve

    def set_roughness(self, roughness):
        """
        Sets the 'roughness' column in the .placements DataFrame

        Parameters
        ----------
            roughness : numeric, iterable
                - If a numeric is given, sets the same roughness values to 
                  all placements
                - If an iterable is given, sets the corresponding roughness 
                  value in the iterable to the placements.
                    - The length of the iterable must match the numbre of 
                      placements 

        Return
        ------
            A reference to the invoking WindWorkflowManager
        """
        self.placements['roughness'] = roughness
        return self

    def estimate_roughness_from_land_cover(self, path, source_type):
        num = gk.raster.interpolateValues(path, self.locs, mode='near')
        self.placements['roughness'] = rk_wind_core.logarithmic_profile.roughness_from_land_cover_classification(
            num, source_type)
        return self

    def logarithmic_projection_of_wind_speeds_to_hub_height(self):
        assert "roughness" in self.placements.columns
        assert hasattr(self, "elevated_wind_speed_height")

        self.sim_data['elevated_wind_speed'] = rk_wind_core.logarithmic_profile.apply_logarithmic_profile_projection(
            self.sim_data['elevated_wind_speed'],
            measured_height=self.elevated_wind_speed_height,
            target_height=self.placements['hub_height'].values,
            roughness=self.placements['roughness'].values
        )
        self.wind_speed_height = self.placements['hub_height'].values

        return self

    def apply_air_density_correction_to_wind_speeds(self):
        assert "surface_air_temperature" in self.sim_data, "surface_air_temperature has not been read from a source"
        assert "surface_pressure" in self.sim_data, "surface_pressure has not been read from a source"
        assert hasattr(self, "elevated_wind_speed_height")

        self.sim_data['elevated_wind_speed'] = rk_wind_core.air_density_adjustment.apply_air_density_adjustment(
            self.sim_data['elevated_wind_speed'],
            pressure=self.sim_data['surface_pressure'],
            temperature=self.sim_data['surface_air_temperature'],
            height=self.wind_speed_height)

        return self

    def convolute_power_curves(self, scaling, base, **kwargs):
        assert hasattr(self, "powerCurveLibrary")

        for key in self.powerCurveLibrary.keys():
            self.powerCurveLibrary[key] = self.powerCurveLibrary[key].convolute_by_guassian(
                scaling=scaling,
                base=base,
                **kwargs
            )

        return self

    def simulate(self):
        gen = np.zeros_like(self.sim_data['elevated_wind_speed'])

        for pckey, pc in self.powerCurveLibrary.items():
            sel = self.placements.powerCurve == pckey
            gen[:, sel] = pc.simulate(self.sim_data['elevated_wind_speed'][:, sel])

        self.sim_data['capacity_factor'] = gen

        return self
