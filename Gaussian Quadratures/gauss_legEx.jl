using Polynomials

function gauss_leg(n)
	# Compute the coefficients of Legendre polynomials using the recursion
	# P(n+1)=(2n+1)P(n)- n H(n-1)
	#          n+1      n+1  
	#
	p   = zeros(3,1)
	p0	= 1
	p1	= [1;0]
	p	= zeros(n,1)
	for i in 1 : n-1
	   p	= (2*i+1)*[p1;0]/(i+1)-i*[0;0;p0]/(i+1)
	   p0	= p1
	   p1	= p
	end
	# Using the polynomial coefficients in p, construct the polynomial pp using the Polynomials package.  This will allow us to use the "roots()" command.
	p   = reverse(p,dims=1)
	#pp	= Poly(p)  # If using Polynomials v0.7
	pp  = Polynomial(p)
	# Compute the polynomial roots:
	x	= sort(roots(pp))
	# Compute the weights imposing that integration is exact for lower order polynomials
	A	= zeros(n,n)
	A[1,:]	= ones(1,n)
	A[2,:]	= x
	for i in 1:n-2
	   A[i+2,:]	= (2*i+1)*x.*A[i+1,:]/(i+1)-i*A[i,:]/(i+1)
	end
	w	= A\[2;zeros(n-1,1)]
	w 	= dropdims(w,dims=2)
	return x, w
end

function gauss_leg(n :: Int64, f, VarArgIn...)
	p   = zeros(3,1)
	p0	= 1
	p1	= [1;0]
	p	= zeros(n,1)
	for i in 1 : n-1
	   p	= (2*i+1)*[p1;0]/(i+1)-i*[0;0;p0]/(i+1)
	   p0	= p1
	   p1	= p
	end
	# Using the polynomial coefficients in p, construct the polynomial pp using the Polynomials package.  This will allow us to use the "roots()" command.
	p   = reverse(p,dims=1)
	#pp	= Poly(p)  # If using Polynomials v0.7
	pp  = Polynomial(p)
	# Compute the polynomial roots:
	x	= sort(roots(pp))
	# Compute the weights imposing that integration is exact for lower order polynomials
	A	= zeros(n,n)
	A[1,:]	= ones(1,n)
	A[2,:]	= x
	for i in 1:n-2;
	   A[i+2,:]	= (2*i+1)*x.*A[i+1,:]/(i+1)-i*A[i,:]/(i+1)
	end
	w	= A\[2;zeros(n-1,1)]
	int = w'*apply(f,x,VarArgIn)
	w 	= dropdims(w,dims=2)
	return x, w, int
end

x, w = gauss_leg(8)