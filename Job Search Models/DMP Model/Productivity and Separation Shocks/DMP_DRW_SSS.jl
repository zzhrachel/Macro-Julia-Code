function StochasticSSDRW(THETA :: Array{Float64,2}, Param :: Parameter_Values)
    ## Simulate for the stochastic steady state.  Only used to set initial value
    # for the unemployment rate in simulation below (could simply choose some 
    # number for the initial unemployment rate that is in [0,1])

    rho     = Param.rho
    beta    = Param.beta
    delta   = Param.delta
    alpha   = Param.alpha
    z       = Param.z
    mA      = Param.mA
    mD      = Param.mD
    b       = Param.b
    PIA     = Param.PIA
    PID     = Param.PID
    nstateA = Param.nstateA
    nstateD = Param.nstateD
    eAt     = Param.eAt
    eDt     = Param.eDt


    x0      = 1
    xss     = fcsolve(DMP_DRWSS,x0,[],Param)
    thss    = exp(xss)
    qss     = 1/((1+thss^alpha)^(1/alpha))
    uss     = delta/(delta + thss*qss)

    ## Now simulate the economy for many periods with no shocks.  That is, set
    # all EXOGENOUS stochastic random variables equal to their mean and simulate for
    # T periods with T being a large number.

    T       = 1000         # Run the simulation for T periods with no shocks.
    ChainA  = ones(T,1)
    statevecA = dropdims(convert(Array{Int64,2},ones(T,1)*ceil(nstateA/2)),dims=2)
    ChainM  = ones(T,1)*alpha
    statevecD = dropdims(convert(Array{Int64,2},ones(T,1)*ceil(nstateD/2)),dims=2)

    uvec    = zeros(T,1)
    thvec   = zeros(T,1)
    vvec    = zeros(T,1)
    wvec    = zeros(T,1)
    fndvec  = zeros(T,1)
    fllvec  = zeros(T,1)
    hvec    = zeros(T,1)
    lpvec   = ChainA
    uvec[1] = uss     # the nonstochastic steady state UR rate for Manovskii-Hagedorn model is 8.01#
                      # The stochastic steady state unemployment rate is 5.32#
    for t in 1 : T
        ut      = uvec[t]
        at      = ChainA[t]
        stateAt = statevecA[t]
        stateDt = statevecD[t]
        tht     = THETA[stateAt,stateDt]
        dt      = eDt[stateDt]
        qt      = 1/((1+tht^alpha)^(1/alpha))
        ht      = ut*tht*qt
        wt      = beta*at + (1-beta)*b + beta*z*tht
        ut1     = ut + dt*(1-ut) - tht*qt*ut
        ut1     = max(ut1,0)
        ut1     = min(ut1,1)
        thvec[t]    = tht
        vvec[t]     = tht*ut 
        fndvec[t]   = tht*qt
        fllvec[t]   = qt
        hvec[t]     = ht
        wvec[t]     = wt
        if t < T
            uvec[t+1]  = ut1
        end
    end

    return uvec[end], vvec[end], hvec[end], fndvec[end], fllvec[end], wvec[end]
end
