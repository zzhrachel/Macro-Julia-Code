using Distributions
using Distributed
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
end

struct SolutionValues
    V :: Array{Float64,2}
    DRH :: Array{Float64,2}

    rhoz :: Float64
    sez :: Float64
    mZ :: Float64
    PIZ :: Array{Float64,2}
    Zt :: Array{Float64,1}

    NGrid :: Array{Float64,}
    NPGrid :: Array{Float64,1}
    nsize :: Int64
    npsize :: Int64
    nstate :: Int64
end


function VPInterpolate(NIntMat :: Array{Float64,2}, ZIntMat :: Array{Float64,2}, V :: Array{Float64,2},
                       NPGrid :: Array{Float64,1}, Zt :: Array{Float64,1}, nsize :: Int64, npsize :: Int64, nstate :: Int64)
    VPIntp = BilinearInterp2D(NIntMat,ZIntMat,reshape(V,nsize,nstate),repeat(NPGrid,1,nstate),repeat(Zt',npsize,1))
    VP = zeros(nsize*nstate,npsize)
    for i in 1 : nstate
        VP[(i-1)*nsize+1:i*nsize,:] = repeat(VPIntp[:,i]',nsize,1)
    end
    return VP
end

function ConstructEVP(PIZ :: Array{Float64,2}, VP :: Array{Float64,2},
                      nstate :: Int64, nsize :: Int64, npsize :: Int64)
    # EVP = zeros(nsize*nstate,npsize)
    # for i in 1 : nstate
    #     for j in 1 : nstate
    #         EVP[(i-1)*nsize+1:i*nsize,:] += PIZ[i,j]*VP[(j-1)*nsize+1:j*nsize,:]
    #     end
    # end
    EMat = @distributed (vcat) for i in 1 : nstate
        ConstructEMat(i,PIZ,VP,nsize,npsize,nstate)
    end
    EVP = zeros(nsize*nstate,npsize)
    k   = 1
    while k <= nstate
        i = convert(Int64,EMat[(k-1)*nsize+1,1])
        EVP[(i-1)*nsize+1:i*nsize,:] = EMat[(k-1)*nsize+1:k*nsize,2:npsize+1]
        k += 1
    end
    return EVP
end

@everywhere function ConstructEMat(i :: Int64, PIZ :: Array{Float64,2}, VP :: Array{Float64,2},
                                  nsize :: Int64, npsize :: Int64, nstate :: Int64)
    Et = zeros(nsize,npsize)
    for j in 1 : nstate
        Et += PIZ[i,j]*VP[(j-1)*nsize+1:j*nsize,:]
    end
    EVPi = hcat(i*ones(nsize,1),Et)
    return EVPi
end

function ConstructDR(PIZ :: Array{Float64,2}, NIntMat :: Array{Float64,2}, ZIntMat :: Array{Float64,2},
                     V :: Array{Float64,2}, NPGrid :: Array{Float64,1}, Zt :: Array{Float64,1},
                     nsize :: Int64, npsize :: Int64, nstate :: Int64, Params :: ParameterValues)
    alpha = Params.alpha
    eta = Params.eta
    rho = Params.rho

    VP  = VPInterpolate(NIntMat, ZIntMat, V, NPGrid, Zt, nsize, npsize, nstate)
    EVP = ConstructEVP(PIZ,VP,nstate, nsize, npsize)

    TV, ind = findmax(ZMat.*NMat.^alpha - 0.5*eta*(HMat./NMat).^2 - w*NMat + rho*EVP,dims=2)
    DR = HMat[ind]
    return DR
end

#----------------------------------------------------------------------------------------------------

#########################
#   Solve the Model:    #
#########################

# Define parameter values:
r = 0.015
w = 0.95
eta = 1
alpha = 0.6
rho = 1/(1+r)
delta = 0.025

Params = ParameterValues(r,w,eta,alpha,rho)

#----------------------------------------------------------------------------------------------------
# Constructing the Markov Chain (Tauchen-Hussey, 1991 following Floden's code (Economics Letters)) :
#----------------------------------------------------------------------------------------------------

rhoz   = 0.85   # Persistance of productivity.
sez    = 0.1    # Volatility of labour productivity
mZ     = 0.0    # mean
nstate = 15      # Number of states for labour productivity

Zgrid, PIZ = rouwenhorst(rhoz,sez,nstate)
Zt  = exp.(Zgrid .+ mZ)

nsize = 151
npsize = 501
nmin  = 0.001
nmax  = 1
NGrid = collect(range(nmin,stop=nmax,length=nsize))
NPGrid = collect(range(nmin,stop=nmax,length=npsize))
NMat  = repeat(NGrid,nstate,npsize)
NPMat = repeat(NPGrid',nsize*nstate,1)
HMat  = NPMat - NMat
ZMat  = kron(Zt,ones(nsize,npsize))

NIntMat = repeat(NGrid,1,nstate)
ZIntMat = repeat(Zt',nsize,1)

V   = zeros(nsize*nstate,1)
tol = 1e-6
crt = 1
while crt > tol
    # Numerically integrate:
    VP  = VPInterpolate(NIntMat, ZIntMat, V, NPGrid, Zt, nsize, npsize, nstate)
    EVP = ConstructEVP(PIZ,VP,nstate, nsize, npsize)

    TV, ind = findmax(ZMat.*NMat.^alpha - 0.5*eta*(HMat./NMat).^2 - w*NMat + rho*EVP,dims=2)
    global crt = maximum(abs.(V - TV)./(1+maximum(abs.(V))))
    global V = copy(TV)
    println(crt)
end
DRH = ConstructDR(PIZ, NIntMat, ZIntMat, V, NPGrid, Zt, nsize, npsize, nstate, Params)

SolutionResults = SolutionValues(V,DRH,rhoz,sez,mZ,PIZ,Zt,NGrid,NPGrid,nsize,npsize,nstate)

saveresults = 1
if saveresults == 1
    save("/Users/jacobwong/Dropbox/Jake's Julia Codes/Post-Version 5.0/Firm Employment Model/LabourConvexAdjModelHiWHiSE.jld",
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
ylabel("TFP",fontdict=font1)
zlabel("Hires",fontdict=font1)
axis([nmin,nmax,minimum(Zt),maximum(Zt)])

# fig = figure("Contour_Decision_Rule")
# contour(repeat(NGrid,1,nstate),repeat(Zt',nsize,1),reshape(DR,nsize,nstate),rstride=1, cstride=1, cmap = ColorMap("jet"))
# clabel()
# xlabel("Employment",fontdict=font1)
# ylabel("Productivity",fontdict=font1)
# zlabel("Hires",fontdict=font1)
# axis([nmin,nmax,minimum(Zt),maximum(Zt)])

println("All Done")
