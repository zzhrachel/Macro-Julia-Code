function DMP_DRW_TRes(x0 :: Array{Float64,1}, Param :: Parameter_Values)
    # This solves the DMP search model by finding the state contingent values for theta (v/u ratio).
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

    TMat    = exp.(x0)
    TMat    = reshape(TMat,nstateA,nstateD)

    q	= 1 ./ ((1 .+ TMat.^(alpha)).^(1/alpha))
    ET  = zeros(nstateA,nstateD)    
    for i in 1 : nstateA
        for j in 1 : nstateD
            tmp = 0;
            for ii in 1 : nstateA
                for jj in 1 : nstateD
                    tmp = tmp + PIA[i,ii]*PID[j,jj]*((1-beta)*(eAt[ii]-b) + (1 .- eDt[jj]-TMat[ii,jj]*q[ii,jj]*beta)*z/q[ii,jj])
                end
            end
            ET[i,j] = tmp
        end
    end

    res = z .- q[:].*rho.*ET[:]
    res	= res./(1+maximum(abs.(res[:])))
    return res
end