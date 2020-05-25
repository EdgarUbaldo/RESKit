from ._util import * #imports everthing from ._util
from ._costModel import * #imports everthing from ._costModel

from scipy.optimize import differential_evolution 
from scipy.stats import exponweib

class _BaselineOnshoreTurbine(dict):
    """
    
    Defines a "baseline" onshore wind turbine with capacity = 42000 kW, hub height = 120 m, and rotor diameter = 136 m that is thought to reflect future trends 
    in wind turbine characteristics (in 2050) according to Ryberg et al. [1]

    References:
    ----------
    [1] 
    
    """

baselineOnshoreTurbine = _BaselineOnshoreTurbine(capacity=4200, hubHeight=120, rotordiam=136, specificPower=289.12230146451577)

def suggestOnshoreTurbine(averageWindspeed, rotordiam=baselineOnshoreTurbine["rotordiam"]):
    """
    Suggest onshore turbine hub height and capacity values based on an average wind speed and the "baseline" onshore turbine as per Ryberg et al. [1]

    Parameters:
    ----------
    averageWindspeed : float or array_like
        Local average wind speed close to or at the hub height

    rotordiam : float or array_like, optional
        Rotor diamter in meters. Default value is 136 
    
    Returns
    -------
    Onshore turbine suggested characteristcs: pandas data frame.
        A pandas data frame with columns hub height in m, specific power in W/m2, and capacity in kW

    Notes
    -------
    Suggestions are given such that with an average wind speed value of 6.7 m/s, a turbine with 4200 kW capacity, 120m hub height, and 136m rotor diameter is chosen
    The specific power (capacity/area of the rotor) is not permited to go less than 180 W/m2 (becase...)
    A minimum hub height to keep 20 m sepatarion distnce beteen the tip of the blade and the floor is maintaied.
    
    References
    -------
    [1] {Ryberg, 2019 #144}


    """
    averageWindspeed = np.array(averageWindspeed) #trasformes the object into a numpy array
    if averageWindspeed.size>1: #if it has more than one element
        multi=True # indicates that many elements are need to be evaluated
        rotordiam = np.array([rotordiam]*averageWindspeed.size) #sets the rotor diameter value (136 m)
    else:   
        multi=False #indicates that there is only one value

    hubHeight = 1.24090975567715489091824565548449754714965820312500*np.exp(-0.84976623*np.log(averageWindspeed)+6.1879937) #sets hub height
    if multi:
        lt20 = hubHeight<(rotordiam/2+20) #kees the 20 m tip-to-floor distnace
        if lt20.any():
            hubHeight[lt20] = rotordiam[lt20]/2 + 20
        # gt200 = hubHeight>200
        # if gt200.any():                                                                                                   ## can wee delete this lt >200 = 200 ??
        #     hubHeight[gt200] = 200
    else:
        if hubHeight<(rotordiam/2+20): hubHeight = rotordiam/2 + 20 
        # if hubHeight>200: hubHeight = 200                                                                                 ## can wee delete this lt >200 = 200 ??
    
    specificPower = 0.90025957072652906809651085495715960860252380371094*np.exp(0.53769024 *np.log(averageWindspeed)+4.74917728) #sets the specific power
    if multi:
        lt180 = specificPower<180
        if lt180.any(): #sets the minimum specific power to 180
            specificPower[lt180] = 180
    else:
        if specificPower<180: specificPower = 180

    capacity = specificPower*np.pi*np.power((rotordiam/2),2)/1000

    output = dict(capacity=capacity, hubHeight=hubHeight, rotordiam=rotordiam, specificPower=specificPower)
    if multi:
        return pd.DataFrame(output)
    else:
        return output



class OptimalTurbine(namedtuple("OptimalTurbine","capacity rotordiam hubHeight opt")):                                                #### I did not use this, right?
    """ 
    
    Defines capacity, hub height and rotor diameter of a baseline turbine that reflects future trends in wind turbine characteristics accoiring to Ryberg et al. [CITE]
    
    """

    
    def __str__(s): #"s" is short for a string?
        out = ""
        out += "Capacity:   %d\n"%int(s.capacity)
        out += "Rotor Diam: %d\n"%int(s.rotordiam)
        out += "Hub Height: %d\n"%int(s.hubHeight)
        out += "LCOE Value: %.5f"%s.opt.fun
        return out
    def __repr__(s): return str(s)

def determineBestTurbine(weibK=2, weibL=7, capacity=(3000,9000), rotordiam=(90,180), hubHeight=(80,200), roughness=0.02, costModel=onshoreTurbineCost, measuredHeight=50, minSpecificCapacity=200, groundClearance=25, tol=1e-5, **kwargs):
    """
    A genetic algorithm to determine the cost-performance optimal onshore turbine characteristics (capacity, rotor diameter, and hub height) for a determined location.
    
    Parameters:
    ----------
        weibK : float 
            Weibull k parameter describing the location's wind speed distribution.

        weibL : float
            Weibull lambda parameter describing the location's wind speed distribution.

        capacity : float or tuple(float,float)
            if float, it will be considered as explicit value and won't be optimized.
            if tuple, it should represent the allowable minimal and maximal capacity values in kW.

        rotordiam : float or tuple(float,float)
            if float, it will be considered as explicit value and won't be optimized.
            if tuple, it should represent the allowable minimal and maximal rotor diamter values in m.

        hubHeight : float or tuple(float,float)
            if float, it will be considered as explicit value and won't be optimized.
            if tuple, it should represent the allowable minimal and maximal hub heights values in m.

        roughness : float
            The roughness length value of the onshore turbine location.

        costModel : callable
            The cost model fuction to be used for calculatiing the onshore turbine's cost. The cost function must accept keyword arguments 
            capacity, rotordiam, and hubHeight and return a single float value.

        measuredHeight : float
            The implied height at which the given windspeed distribution was determined.

        minSpecificCapacity : float or None
            The minimal specific-capacity value to allow during the optimization
            if None, it will imply no minimum speficic capacity limit.

        groundClearance : float
            The minimal height allowed between ground and the rotor tip in m.

        tol : float, optional
            The tolerance to use during the optimization. Default is 1e-5. See scipy.optimize.differential_evolution for more information.

        **kwargs: key word arguments, optional
            All other kwargs are passed on to scipy.optimize.differential_evolution

    Returns
    -------
        **I NEED TO DIG DEEPER IN THIS**
    
    Notes:
    ------
        A synthetic turbine power curve is always generated according to the given capacity and rotor diameter
        *I NEED TO PROVIDE MORE DETAILS**
    -------

    """
    ws = np.linspace(0,40,4000)
    dws = ws[1]-ws[0]
    _s = np.log(measuredHeight/roughness)

    # Determine unpacking, boundary, and finalization structure
    if isinstance(capacity,tuple) and isinstance(rotordiam,tuple) and isinstance(hubHeight,tuple):
        unpack = lambda x: (x[0], x[1], x[2])
        bounds = [capacity, rotordiam, hubHeight, ]
        finalize = lambda x: OptimalTurbine(x.x[0], x.x[1], x.x[2], x)
    elif not isinstance(capacity,tuple) and isinstance(rotordiam,tuple) and isinstance(hubHeight,tuple):
        unpack = lambda x: (capacity, x[0], x[1])
        bounds = [rotordiam, hubHeight, ]
        finalize = lambda x: OptimalTurbine(capacity, x.x[0], x.x[1], x)
    elif isinstance(capacity,tuple) and not isinstance(rotordiam,tuple) and isinstance(hubHeight,tuple):
        unpack = lambda x: (x[0], rotordiam, x[1])
        bounds = [capacity, hubHeight, ]
        finalize = lambda x: OptimalTurbine(x.x[0], rotordiam, x.x[1], x)
    elif isinstance(capacity,tuple) and isinstance(rotordiam,tuple) and not isinstance(hubHeight,tuple):
        unpack = lambda x: (x[0], x[1], hubHeight)
        bounds = [capacity, rotordiam, ]
        finalize = lambda x: OptimalTurbine(x.x[0], x.x[1], hubHeight, x)
    elif not isinstance(capacity,tuple) and not isinstance(rotordiam,tuple) and isinstance(hubHeight,tuple):
        unpack = lambda x: (capacity, rotordiam, x[0])
        bounds = [hubHeight, ]
        finalize = lambda x: OptimalTurbine(capacity, rotordiam, x.x[0], x)
    elif not isinstance(capacity,tuple) and isinstance(rotordiam,tuple) and not isinstance(hubHeight,tuple):
        unpack = lambda x: (capacity, x[0], hubHeight)
        bounds = [rotordiam, ]
        finalize = lambda x: OptimalTurbine(capacity, x.x[0], hubHeight, x)
    elif isinstance(capacity,tuple) and not isinstance(rotordiam,tuple) and not isinstance(hubHeight,tuple):
        unpack = lambda x: (x[0], rotordiam, hubHeight)
        bounds = [capacity, ]
        finalize = lambda x: OptimalTurbine(x.x[0], rotordiam, hubHeight, x)
    else:
        raise RuntimeError("Something is wrong...")

    # Define scoring function
    def score(x):
        """ 
    
        DOCSTRING NEEDED
        
        """
        c,r,h = unpack(x)
        s = np.log(h/roughness)/_s
        pdf = exponweib.pdf(ws, a=1, c=weibK, loc=0, scale=weibL*s)
        
        pc = SyntheticPowerCurve(capacity=c, rotordiam=r)
        cf = np.interp(ws, pc.ws, pc.cf)
        
        expectedCapFac = (cf*pdf).sum()*dws
        capex = costModel(capacity=c, hubHeight=h, rotordiam=r)
        lcoe = simpleLCOE(capex, expectedCapFac*8760*c)
        
        # Dissuade against too low specific-capacity values
        if not minSpecificCapacity is None:
            specificCapacity = 1000*c/(np.pi*r*r/4)
            if specificCapacity<minSpecificCapacity:
                lcoe += np.power(minSpecificCapacity-specificCapacity,3)

        # Dissuade against too-low hub height compared to the rotor diameter
        tmp = (groundClearance+r/2)-h
        if tmp>0: lcoe += np.power(tmp,3)

        # Done!
        return lcoe

    res = differential_evolution(score, bounds=bounds, tol=tol, **kwargs)
    return finalize(res)
