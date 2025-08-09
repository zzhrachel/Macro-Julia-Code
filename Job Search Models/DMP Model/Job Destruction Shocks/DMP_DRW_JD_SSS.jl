function StochasticSSDRW(THETA :: Array{Float64,1}, Param :: Parameter_Values)
    ## Simulate for the stochastic steady state.  Only used to set initial value
    # for the unemployment rate in simulation below (could simply choose some 
    # number for the initial unemployment rate that is in [0,1])

    rho     = Param.rho
    beta    = Param.beta
    delta   = Param.delta
    alpha   = Param.alpha
    z       = Param.z
    b       = Param.b
    eAt     = Param.eAt
    PID     = Param.PID
    nstateD = Param.nstateD
    eDt     = Param.eDt


    x0      = exp(0)
    xxss    = fcsolve(DMP_DRW_SS,x0,[],Param)
    thss    = exp(xxss)
    qss     = 1/((1+thss^-alpha)^(1/alpha))
    uss     = delta/(delta + thss*qss)

    ## Now simulate the economy for many periods with no shocks.  That is, set
    # all EXOGENOUS stochastic random variables equal to their mean and simulate for
    # T periods with T being a large number.

    T       = 1000         # Run the simulation for T periods with no shocks.
    ChainD  = ones(T,1)*delta
    statevecD = dropdims(convert(Array{Int64,2},ones(T,1)*ceil(nstateD/2)),dims=2)

    uvec    = zeros(T,1)
    thvec   = zeros(T,1)
    vvec    = zeros(T,1)
    wvec    = zeros(T,1)
    fndvec  = zeros(T,1)
    fllvec  = zeros(T,1)
    hvec    = zeros(T,1)
    dvec    = ChainD
    display(ChainD)
    uvec[1] = uss     # the nonstochastic steady state UR rate for Manovskii-Hagedorn model is 8.01#
                      # The stochastic steady state unemployment rate is 5.32#
    for t in 1 : T
        ut      = uvec[t]
        dt      = ChainD[t]
        stateDt = statevecD[t]
        tht     = THETA[stateDt]
        qt      = 1/((1+tht^alpha)^(1/alpha))
        ht      = ut*tht*qt
        wt      = beta*eAt + (1-beta)*b + beta*z*tht
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

    return uvec[end], vvec[end], hvec[end], fndvec[end], fllvec[end]
end
