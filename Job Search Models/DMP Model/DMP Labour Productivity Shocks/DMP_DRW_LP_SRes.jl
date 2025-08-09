function DMP_DRW_SRes(x0 :: Array{Float64,1}, Param :: Parameter_Values);
    # Solve for equilibrium of the DMP Model by solving for the joint surplus 
    # state-by-state.
    rho     = Param.rho
    beta    = Param.beta
    delta   = Param.delta
    alpha   = Param.alpha
    z       = Param.z
    mA      = Param.mA
    b       = Param.b
    PIA     = Param.PIA
    nstateA = Param.nstateA
    eAt     = Param.eAt
    
    SMat    = exp.(x0)  # Make sure that the surplus of a job is strictly positive
    
    ES      = zeros(nstateA,1)
    for i in 1 : nstateA
        tmp = 0
        for ii in 1 : nstateA
            tmp = tmp + PIA[i,ii]*SMat[ii]
        end
        ES[i] = tmp
    end
    THETA   = (((rho*(1-beta)*ES)/z).^(alpha).-1).^(1/alpha)
    jfnd    = 1 ./ ((1 .+ THETA.^(-alpha)).^(1/alpha))
    TS      = eAt .- b + rho*(1 .- beta*jfnd .- delta).*ES
    res = SMat[:] - TS[:]
    return res
end