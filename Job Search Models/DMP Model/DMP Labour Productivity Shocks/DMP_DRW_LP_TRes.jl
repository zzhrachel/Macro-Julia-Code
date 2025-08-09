function DMP_DRW_TRes(x0 :: Array{Float64,1}, Param :: Parameter_Values)
    # This solves the DMP search model by finding the state contingent values for theta (v/u ratio).
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

    TMat    = exp.(x0)
    TMat    = reshape(TMat,nstateA,1)

    q   = 1 ./ ((1 .+ TMat.^(alpha)).^(1/alpha))
    ET  = zeros(nstateA,1)    
    for i in 1 : nstateA
        tmp = 0;
        for ii in 1 : nstateA
            tmp = tmp + PIA[i,ii]*((1-beta)*(eAt[ii]-b) + (1 - delta - TMat[ii]*q[ii]*beta)*z/q[ii])
        end
        ET[i] = tmp
    end
    res = z .- q[:].*rho.*ET[:]
    res = res./(1+maximum(abs.(res[:])))
    return res
end
