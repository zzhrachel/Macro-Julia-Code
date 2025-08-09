function DMP_DRWRes(x0 :: Array{Float64,1}, Param :: Parameter_Values);
    # Solve for equilibrium of the DMP Model by solving for the joint surplus state-by-state.
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

    SMat    = exp.(x0)
    SMat    = reshape(SMat,nstateA,nstateD)

    ES  = zeros(nstateA,nstateD)
    for i in 1 : nstateA
        for j in 1 : nstateD
            tmp = 0
            for ii in 1 : nstateA
                for jj in 1 : nstateD
                    tmp = tmp + PIA[i,ii]*PID[j,jj]*SMat[ii,jj]
                end
            end
            ES[i,j] = tmp
        end
    end
    THETA   = (((rho*(1-beta)*ES)/z).^(alpha)-1).^(1/alpha)
    jfnd    = 1 ./ ((1+THETA.^(-alpha)).^(1/alpha))
    TS      = repmat(eAt,1,nstateD) .- b + rho*(1 .- beta*jfnd - repmat(eDt',nstateA,1)).*ES

    res = SMat[:] - TS[:]
    return res
end