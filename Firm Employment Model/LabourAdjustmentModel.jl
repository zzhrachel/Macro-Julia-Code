using Distributions
using PyPlot

include("lgwt.jl")
include("BilinearInterp2D.jl")

function transform_e_to_z(evec :: Array{Float64,1},zmin :: Float64, zmax :: Float64)
    zn = 0.5*(evec+1)*(zmax-zmin) + zmin
    return zn
end

# Define parameter values:
r = 0.01
w = 0.6
p = 0.05
zm = 1
eta = 0.1
alpha = 0.6
rho = 1/(1+r)

mu  = 0.5
sg  = 0.1
A   = ((1-mu)*mu^2)/sg - mu
B   = A*(1-mu)/mu

zmin = -0.5
zmax = 0.5
nsize = 101
npsize = 501
NGrid = collect(linspace(0.01,1,nsize))
NPGrid = collect(linspace(0.01,1,npsize))
NPMat = repmat(NPGrid',nsize,1)
zsize = 51
ZGrid = collect(linspace(zmin,zmax,zsize))

inodes = 12
x,q  = lgwt(inodes,-1,1)
# Map the domain of x, [-1,1], into the domain of the beta distribution, [0,1] using a linear change-in-varibles:
evec = 0.5*(x+1)
# Map the epsilon integration nodes into nodes in [zmin,zmax].
zn   = transform_e_to_z(x,zmin,zmax)
ZiMat = repmat(zn',npsize,1)
vsize = nsize*zsize
IntVec = 1/(2*beta(A,B))*q.*evec.^(A-1).*(1-evec).^(B-1)

NMat  = repmat(NGrid,1,zsize)
ZMat  = repmat(ZGrid',nsize,1)
NPiMat = repmat(NPGrid,1,inodes)
BigN  = repmat(repmat(NGrid,zsize,1),1,npsize)
BigZ  = repmat(kron(ZGrid,ones(nsize,1)),1,npsize)
BigH  = repmat(NPMat - repmat(NGrid,1,npsize),zsize,1)
BigNP = repmat(NPMat,zsize,1)


# V   = zeros(vsize,1)
# tol = 1e-6
# crt = 1
# while crt > tol
#     # Numerically integrate:
#     VNPMat  = BilinearInterp2D(NMat,ZMat,reshape(V,nsize,zsize),NPiMat,ZiMat)
#     #VNMat   = BilinearInterp2D(NMat,ZMat,reshape(V,nsize,zsize),BigNP,ZMat)
#     iVNPMat = VNPMat*IntVec
#     EVNPMat = repmat(iVNPMat',nsize*zsize,1)

#     TV, ind = findmax(zm*exp.(BigZ).*BigN.^alpha - 0.5*eta*(BigH./BigN).^2 - w*BigN + rho*EVNPMat,2)
#     crt = maximum(abs.(V - TV)./(1+maximum(abs.(V))))
#     V = copy(TV)
#     println(crt)
# end

# # Back out hiring decision rule:
# # Numerically integrate:
# VNPMat  = BilinearInterp2D(NMat,ZMat,reshape(V,nsize,zsize),NPiMat,ZiMat)
# iVNPMat = VNPMat*IntVec
# EVNPMat = repmat(iVNPMat',nsize*zsize,1)
# TV, ind = findmax(zm*exp.(BigZ).*BigN.^alpha - 0.5*eta*(BigH./BigN).^2 - w*BigN + rho*EVNPMat,2)
# DRH = BigH[ind]
# DRH = reshape(DRH,nsize,zsize)


# # Plot Hiring Decision Rule:
# font1 = Dict("family"=>"serif",
# # "name" => "times",
# "name" => "Times New Roman",
# #"color"=>"darkred",
# #"weight"=>"normal",
# "size"=>14)
# #xlabel("Time",fontdict=font1)

# font2 = Dict("family"=>"serif",
# # "name" => "times",
# "name" => "Times New Roman",
# #"color"=>"darkred",
# #"weight"=>"normal",
# "size"=>16)
# #xlabel("Time",fontdict=font1)

# font3 = Dict("family"=>"serif",
# # "name" => "times",
# "name" => "Times New Roman",
# #"color"=>"darkred",
# #"weight"=>"normal",
# "size"=>12)
# #xlabel("Time",fontdict=font1)

# fig = figure("Hiring_Decision_Rule")
# surf(NMat,ZMat,DRH,rstride=1, cstride=1, cmap = ColorMap("jet"))
# # surf(repmat(midDistBins,1,MsizeC-1),repmat(midMeanBins',MsizeD-1,1),UMean3DMat, rstride=1, cstride=1, cmap = ColorMap("jet"))
