function DMP_DRW_LP_Simulate(nsim :: Int64, Brn :: Int64, THETA :: Array{Float64},
                             SSSvec  :: Array{Float64,1}, Param :: Parameter_Values)

    rho     = Param.rho
    beta    = Param.beta
    delta   = Param.delta
    alpha   = Param.alpha
    z       = Param.z
    b       = Param.b
    mA      = Param.mA
    PIA     = Param.PIA
    nstateA = Param.nstateA
    eAt     = Param.eAt

    usss = SSSvec[1]

    ## Simulating the Model for Many Periods
    #------------------------
    # Simulating the Model : 
    #------------------------

    # First simulate a sequence for the TFP process.  See class notes on 
    # Simulating Markov Chains.

    cumPIA  = zeros(nstateA,nstateA)
    for k   = 1 : nstateA
        tmp = 0
        for kk = 1 : nstateA
            tmp          = tmp + PIA[k,kk]
            cumPIA[k,kk] = tmp
        end
    end

    #-----------------------------------------
    # Simulate a draw from the Markov Chains:
    #-----------------------------------------
    sA0     = ceil(nstateA/2)  # Set the initial state to equal the state with the median of the labour 
                               # productivity process.  (There are nstateA states for labour productivity)
    Brn     = 1000             # Number of Burn-in Periods to discard in simulation.
    T       = Brn + nsim       # Total number of periods to simulate.
    Random.seed!(27213)     # Store the seed for the vector of random numbers to be used in the simulations
    dU      = Uniform(0,1)
    p       = rand(dU,T)    # Draw T realizations of a random variable that is uniformly distributed over the [0,1] interval.  These
                            # can be treated as probabilities.
    drw     = convert(Array{Int64,1},dropdims(zeros(T,1),dims=2))    # The j's will be the TFP realizations.
    drw[1]  = sA0
    for k in 2 : T
        drw[k]    = minimum(findall(cumPIA[drw[k-1],:] .> p[k]))
    end
    ChainA  = eAt[drw]
    # Now construct a vector indicating the state for TFP in each period
    statevecA   = zeros(T,1);
    for kk = 1 : T
        for k = 1 : nstateA
            if ChainA[kk] == eAt[k]
                statevecA[kk] = k
            end
        end            
    end
    statevecA = convert(Array{Int64,1},dropdims(statevecA,dims=2))

    #-----------------------
    # Simulate the economy:
    #-----------------------
    uvec    = zeros(T,1)
    thvec   = zeros(T,1)
    vvec    = zeros(T,1)
    wvec    = zeros(T,1)
    fndvec  = zeros(T,1)
    fllvec  = zeros(T,1)
    hvec    = zeros(T,1)
    uvec[1] = SSSvec[1]
    for t = 1 : T
        ut      = uvec[t]
        at      = ChainA[t]
        stateAt = statevecA[t]
        tht     = THETA[stateAt]
        qt      = 1/((1+tht^alpha)^(1/alpha));
        ht      = ut*tht*qt;
        wt      = beta*at + (1-beta)*b + beta*z*tht;
        ut1     = ut + delta*(1-ut) - tht*qt*ut;
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
    ## Save simulation results (after dropping burn-in periods)
    return uvec[Brn+1:T], vvec[Brn+1:T], thvec[Brn+1:T], hvec[Brn+1:T], fndvec[Brn+1:T], fllvec[Brn+1:T], 
           wvec[Brn+1:T], ChainA[Brn+1:T]
end
