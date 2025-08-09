# This code uses value function iteration to solve Hopenhayn and Nicolini's model from their JPE paper.
# It is a direct translation from Matlab to Julia.  Note that it is much slower than the Matlab version
# because Matlab is optimized to work with matrix algebra while Julia is optimized for looping.  In order
# to make a fair speed comparison, this code should be rewritten to run using loops.

include("/Users/jacobwong/Dropbox/Macro IV Computer Codes/Julia Codes/Post-Version 1.0/Useful Functions/LinearInterp1D.jl")
using PythonPlot
using LaTeXStrings

struct ParameterValues
	beta :: Float64
	r :: Float64
	sigma :: Real
	w :: Real
	Aut :: Real
	nV :: Int64
	vma :: Real
	nVp :: Int64
end

function CRRA_util(x :: Real, sigma :: Real)
	u = x^(1-sigma)/(1-sigma)
	return u
end

function CRRA_util(x :: Array, sigma :: Real)
	u = x.^(1-sigma)./(1-sigma)
	return u
end

function log_util(x :: Real, sigma :: Real)
	u = log(x)
	return u
end

function log_util(x :: Array, sigma :: Real)
	u = log(x)
	return u
end

function VFI(V :: Array, VMat :: Array, VpInd :: Array, VpMat :: Array, C :: Array,
			 a :: Array, b :: Array, Param :: ParameterValues)
	beta = Param.beta
	r  = Param.r
	nV = Param.nV
	crt = 1
	tol = 1e-8
	while crt > tol
		Cp  = LinearInterp1D(V,C,Vp')
		Cp  = Cp'
		TC, k = findmin(max.(0,b + beta*exp.(-r*a).*repeat(Cp,nV,1)),dims=2)   # k is the index of the optimal promised utility and associated b
		crt = maximum(abs.(C - TC))/(1 + maximum(abs.(C - TC)))
	    println(crt)
		C       = TC
		VMat    = VpMat[k]
	    VpInd   = k
	    println(crt)
	end
	return VMat, VpInd, C
end

# Parameter values
beta    = 0.999
r       = 3.431409393627864e-004
sigma   = 0.5
w       = 100
W       = CRRA_util(w,sigma)/(1-beta)
Aut     = 16759  # Can also try Aut = 16758.6982314168

# Set up grid for V :
nV  = 1000
vmi = Aut
vma = 17000
V   = collect(range(vmi,stop=vma,length=nV))

# Set up grid for Vp :
nVp = 1000
Vp  = collect(range(vmi,stop=vma,length=nVp))'

Param = ParameterValues(beta,r,sigma,w,Aut,nV,vma,nVp)

a   = max.(0,log.(beta*r*(W .- repeat(Vp,nV,1)))/r)
pa  = 1 .- exp.(-r*a);  # Job finding probability given At.
VpMat = repeat(Vp,nV,1)

b   = max.(0,((1-sigma)*(repeat(V,1,nVp) + a - beta*(1 .- exp.(-r*a))*W - beta*exp.(-r*a).*repeat(Vp,nV,1))).^(1/(1-sigma)))

# Initiate Government's value function :
C       = zeros(nV,1)
VMat    = zeros(nV,1)
VpInd   = zeros(nV,1)

VMat, VpInd, C = VFI(V, VMat, VpInd, VpMat, C, a, b, Param)

AMat    = zeros(nV,1)
BMat    = zeros(nV,1)
for i = 1 : nV    
    AMat[i] = a[VpInd[i]]
    BMat[i] = b[VpInd[i]]
    #AMat(i) = LinearInterp1D(Vp',a[i,:]',VMat[i]);
    #BMat(i) = LinearInterp1D(Vp',b[i,:]',VMat[i]);
end

font1 = Dict("family"=>"serif",
"name" => "times",
#"color"=>"darkred",
#"weight"=>"normal",
"size"=>12)
#xlabel("Time",fontdict=font1)

font2 = Dict("family"=>"serif",
"name" => "times",
#"color"=>"darkred",
#"weight"=>"normal",
"size"=>14)
#xlabel("Time",fontdict=font1)


fig = figure(1)
ax  = gca
subplot(2,2,1)
plot(V,AMat,color="b",linestyle="-")
xlabel(L"V_{t}",fontdict=font1)
ylabel(L"a_{t}",fontdict=font1)
axis([minimum(V),maximum(V), minimum(AMat), maximum(AMat)])
title("Worker Search Effort",fontdict=font2)
subplot(2,2,2)
plot(V,BMat./w,color="b",linestyle="-")
xlabel(L"V_{t}",fontdict=font1)
ylabel(L"b_{t}/w",fontdict=font1)
title("Replacement Ratio",fontdict=font2)
subplot(2,2,3)
plot(V,VMat,color="b",linestyle="-",label="Promised Utility")
plot(V,V,color="r",linestyle="-",label="V")
xlabel(L"V_{t}",fontdict=font1)
axis([minimum(V),maximum(V),minimum(VMat),maximum(VMat)])
ylabel(L"V_{t+1}",fontdict=font1)
legend(loc=2)
title("Promised Continuation Utility",fontdict=font2)
subplot(2,2,4)
plot(V,C,color="b",linestyle="-")
xlabel(L"V_{t}",fontdict=font1)
axis([vmi,vma,minimum(C),maximum(C)])
ylabel(L"C(V_{t})",fontdict=font1)
title("Expected Cost to Government",fontdict=font2)
