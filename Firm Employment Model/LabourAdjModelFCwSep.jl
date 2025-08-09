using Distributions
using PythonPlot
using HDF5, JLD

include("rouwenhorst.jl")
include("BilinearInterp2D.jl")

struct ParameterValues
    r :: Float64
    w :: Float64
    eta :: Float64
    alpha :: Float64
    rho :: Float64
    phi :: Float64
    delta :: Float64
end

struct SolutionValues
    V :: Array{Float64,2}
    J :: Array{Float64,2}
    DRH :: Array{Float64,2}

    rhoz :: Float64
    sez :: Float64
    mZ :: Float64
    PIZ :: Array{Float64,2}
    Zt :: Array{Float64,1}

    NVec :: Array{Float64,2}
    ZVec :: Array{Float64,2}
    NPVGrid :: Array{Float64,1}
    NPJGrid :: Array{Float64,1}
    nsize :: Int64
    npsize :: Int64
    nstate :: Int64
end

#----------------------------------------------------------------------------------------------------
# Define functions:
#----------------------------------------------------------------------------------------------------

function NoHiringInterpolate(NIntMat :: Array{Float64,2}, ZIntMat :: Array{Float64,2}, V :: Array{Float64,2}, J :: Array{Float64,2},
                             NPVGrid :: Array{Float64,1}, Zt :: Array{Float64,1}, nsize :: Int64, npsize :: Int64, nstate :: Int64)
    VPIntp = BilinearInterp2D(NIntMat,ZIntMat,reshape(V,nsize,nstate),repeat(NPVGrid,1,nstate),repeat(Zt',nsize,1))
    JPIntp = BilinearInterp2D(NIntMat,ZIntMat,reshape(J,nsize,nstate),repeat(NPVGrid,1,nstate),repeat(Zt',nsize,1))

    VP = reshape(VPIntp,nsize*nstate,1)
    JP = reshape(JPIntp,nsize*nstate,1)
    return VP, JP
end

function HiringInterpolate(NIntMat :: Array{Float64,2}, ZIntMat :: Array{Float64,2}, V :: Array{Float64,2}, J :: Array{Float64,2},
                           NPJGrid :: Array{Float64,1}, Zt :: Array{Float64,1}, nsize :: Int64, npsize :: Int64, nstate :: Int64)
    VPIntp = BilinearInterp2D(NIntMat,ZIntMat,reshape(V,nsize,nstate),repeat(NPJGrid,1,nstate),repeat(Zt',npsize,1))
    JPIntp = BilinearInterp2D(NIntMat,ZIntMat,reshape(J,nsize,nstate),repeat(NPJGrid,1,nstate),repeat(Zt',npsize,1))
    VP = zeros(nsize*nstate,npsize)
    JP = zeros(nsize*nstate,npsize)
    for i in 1 : nstate
        VP[(i-1)*nsize+1:i*nsize,:] = repeat(VPIntp[:,i]',nsize,1)
        JP[(i-1)*nsize+1:i*nsize,:] = repeat(JPIntp[:,i]',nsize,1)
    end
    return VP, JP
end

function ConstructEVP(PIZ :: Array{Float64,2}, VP :: Array{Float64,2}, JP :: Array{Float64,2}, nstate :: Int64, nsize :: Int64)
    EVP = zeros(nsize*nstate,1)
    for i in 1 : nstate
        for j in 1 : nstate
            EVP[(i-1)*nsize+1:i*nsize,:] += PIZ[i,j]*max.(VP[(j-1)*nsize+1:j*nsize,:],JP[(j-1)*nsize+1:j*nsize,:])
        end
    end
    return EVP
end

function ConstructEJP(PIZ :: Array{Float64,2}, VP :: Array{Float64,2}, JP :: Array{Float64,2},
                      nstate :: Int64, nsize :: Int64, npsize :: Int64)
    # Same as ConstructEVP
    EJP = zeros(nsize*nstate,npsize)
    for i in 1 : nstate
        for j in 1 : nstate
            EJP[(i-1)*nsize+1:i*nsize,:] += PIZ[i,j]*max.(VP[(j-1)*nsize+1:j*nsize,:],JP[(j-1)*nsize+1:j*nsize,:])
        end
    end
    return EJP
end


function ConstructDRH(PIZ :: Array{Float64,2}, NIntMat :: Array{Float64,2}, ZIntMat :: Array{Float64,2},
                     V :: Array{Float64,2}, J :: Array{Float64,2}, NPVGrid :: Array{Float64,1}, NPJGrid :: Array{Float64,1}, 
                     NVec :: Array{Float64,2}, ZVec :: Array{Float64,2}, Zt :: Array{Float64,1},
                     nsize :: Int64, npsize :: Int64, nstate :: Int64, Params :: ParameterValues)
    alpha = Params.alpha
    eta = Params.eta
    rho = Params.rho
    phi = Params.phi

    # Numerically integrate:
    VPNoH, JPNoH = NoHiringInterpolate(NIntMat, ZIntMat, V, J, NPVGrid, Zt, nsize, npsize, nstate)
    VPH, JPH = HiringInterpolate(NIntMat, ZIntMat, V, J, NPJGrid, Zt, nsize, npsize, nstate)
    
    EVP = ConstructEVP(PIZ, VPNoH, JPNoH, nstate, nsize)
    EJP = ConstructEJP(PIZ, VPH, JPH, nstate, nsize, npsize)

    TV  = ZVec.*NVec.^alpha - w*NVec + rho*EVP
    TJ, ind = findmax(ZMat.*NMat.^alpha - 0.5*eta*(HMat./NMat).^2 - w*NMat .- phi + rho*EJP,dims=2)

    DRH = HMat[ind]
    for i in 1 : nsize*nstate
        if TV[i] > TJ[i]
            DRH[i] = 0
        end
    end
    return DRH
end

#----------------------------------------------------------------------------------------------------

#########################
#   Solve the Model:    #
#########################

# Define parameter values:
r = 0.015
w = 0.95      # High wage setting w = 0.95.  Low wage setting w = 0.75
eta = 1
alpha = 0.6
rho = 1/(1+r)
phi = 0.01
delta = 0.025

Params = ParameterValues(r,w,eta,alpha,rho,phi,delta)

#----------------------------------------------------------------------------------------------------
# Constructing the Markov Chain (Tauchen-Hussey, 1991 following Floden's code (Economics Letters)) :
#----------------------------------------------------------------------------------------------------

rhoz   = 0.85   # Persistance of productivity.
sez    = 0.1    # Volatility of labour productivity.  High volatility sez = 0.01, low volatiliy sez = 0.001
mZ     = 0.0    # mean
nstate = 15     # Number of states for labour productivity

Zgrid, PIZ = rouwenhorst(rhoz,sez,nstate)
Zt  = exp.(Zgrid .+ mZ)

nsize = 151
npsize = 501
nmin  = 0.001
nmax  = 1
NGrid = collect(range(nmin,stop=nmax,length=nsize))
NPVGrid = dropdims(max.((1-delta)*NGrid,nmin*ones(nsize,1)),dims=2)
NPJGrid = collect(range(nmin,stop=nmax,length=npsize))
NVec  = repeat(NGrid,nstate,1)
NMat  = repeat(NPVGrid,nstate,npsize)
NPJMat = repeat(NPJGrid',nsize*nstate,1)
HMat  = NPJMat - (1-delta)*NMat
ZVec  = kron(Zt,ones(nsize,1))
ZMat  = kron(Zt,ones(nsize,npsize))

NIntMat = repeat(NGrid,1,nstate)
ZIntMat = repeat(Zt',nsize,1)

load_soln = 0
if load_soln == 1
    # This only works if the outer bounds of Zt are greater than that of SolutionResults.Zt and similarly for NGrid.
    # This means that it only works if nstate is smaller than that of the saved solution model AND the Ngrid bounds
    # are not expanded.
    SolutionResults = load("/Users/jacobwong/Dropbox/Jake's Julia Codes/Post-Version 5.0/Firm Employment Model/LabourAdjModelFCwSepUberWHiSE.jld",
                                "SolutionResults")

    nstate_s = SolutionResults.nstate
    nsize_s = SolutionResults.nsize
    NVec_s = SolutionResults.NVec
    ZVec_s = SolutionResults.ZVec
    Vs  = reshape(SolutionResults.V,nsize_s,nstate_s)
    Js  = reshape(SolutionResults.J,nsize_s,nstate_s)

    V   = BilinearInterp2D(reshape(NVec_s,nsize_s,nstate_s),reshape(ZVec_s,nsize_s,nstate_s),Vs,reshape(NVec,nsize,nstate),reshape(ZVec,nsize,nstate))
    J   = BilinearInterp2D(reshape(NVec_s,nsize_s,nstate_s),reshape(ZVec_s,nsize_s,nstate_s),Js,reshape(NVec,nsize,nstate),reshape(ZVec,nsize,nstate))

    V   = reshape(V,nsize*nstate,1)
    J   = reshape(J,nsize*nstate,1)
else
    V   = ones(nsize*nstate,1)
    J   = ones(nsize*nstate,1)
end

tol = 1e-6
crt = 1
while crt > tol
    # Numerically integrate:
    VPNoH, JPNoH = NoHiringInterpolate(NIntMat, ZIntMat, V, J, NPVGrid, Zt, nsize, npsize, nstate)
    VPH, JPH = HiringInterpolate(NIntMat, ZIntMat, V, J, NPJGrid, Zt, nsize, npsize, nstate)
    
    EVP = ConstructEVP(PIZ, VPNoH, JPNoH, nstate, nsize)
    EJP = ConstructEJP(PIZ, VPH, JPH, nstate, nsize, npsize)

    TV  = ZVec.*NVec.^alpha - w*NVec + rho*EVP
    TJ, ind = findmax(ZMat.*NMat.^alpha - 0.5*eta*(HMat./NMat).^2 - w*NMat .- phi + rho*EJP,dims=2)
    res = [V - TV ; J - TJ]
    global crt = maximum(abs.(res)./(1+maximum(abs.(res))))
    global V = copy(TV)
    global J = copy(TJ)
    println(crt)
end
DRH = ConstructDRH(PIZ, NIntMat, ZIntMat, V, J, NPVGrid, NPJGrid, NVec, ZVec, Zt, nsize, npsize, nstate, Params)


SolutionResults = SolutionValues(V,J,DRH,rhoz,sez,mZ,PIZ,Zt,NVec,ZVec,NPVGrid,NPJGrid,nsize,npsize,nstate)

saveresults = 1
if saveresults == 1
    save("/Users/jacobwong/Dropbox/Jake's Julia Codes/Post-Version 5.0/Firm Employment Model/LabourAdjModelFCwSepUberWHiSE.jld",
            "SolutionResults",SolutionResults,"Params",Params)
end

# Plot Hiring Decision Rule:
font1 = Dict("family"=>"serif",
# "name" => "times",
"name" => "Times New Roman",
#"color"=>"darkred",
#"weight"=>"normal",
"size"=>14)
#xlabel("Time",fontdict=font1)

font2 = Dict("family"=>"serif",
# "name" => "times",
"name" => "Times New Roman",
#"color"=>"darkred",
#"weight"=>"normal",
"size"=>16)
#xlabel("Time",fontdict=font1)

font3 = Dict("family"=>"serif",
# "name" => "times",
"name" => "Times New Roman",
#"color"=>"darkred",
#"weight"=>"normal",
"size"=>12)
#xlabel("Time",fontdict=font1)

fig = figure("Surface_Decision_Rule", figsize=(8*1.05,14*1.05))
surf(repeat(NGrid,1,nstate),repeat(Zt',nsize,1),reshape(DRH,nsize,nstate),rstride=1, cstride=1, cmap = ColorMap("jet"))
xlabel("Employment",fontdict=font1)
ylabel("Productivity",fontdict=font1)
zlabel("Hires",fontdict=font1)
axis([nmin,nmax,minimum(Zt),maximum(Zt)])

println("All Done")

