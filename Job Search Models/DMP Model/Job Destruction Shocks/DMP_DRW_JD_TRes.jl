function DMP_DRW_JD_Res(x0 :: Array{Float64,1}, Param :: Parameter_Values)
    # This solves the DMP search model by finding the state contingent values for theta (v/u ratio).
    rho     = Param.rho
    beta    = Param.beta
    alpha   = Param.alpha
    z       = Param.z
    b       = Param.b
    PID     = Param.PID
    nstateD = Param.nstateD
    eDt     = Param.eDt

    TMat    = exp.(x0)
    TMat    = reshape(TMat,nstateD,1)

    q   = 1 ./ ((1 .+ TMat.^(alpha)).^(1/alpha))
    ET  = zeros(nstateD,1)    
    for i in 1 : nstateD
        tmp = 0;
        for ii in 1 : nstateD
            tmp = tmp + PID[i,ii]*((1-beta)*(eAt-b) + (1 - eDt[i] - TMat[ii]*q[ii]*beta)*z/q[ii])
        end
        ET[i] = tmp
    end
    res = z .- q[:].*rho.*ET[:]
    res = res./(1+maximum(abs.(res[:])))
    return res
end
