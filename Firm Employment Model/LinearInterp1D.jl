function LinearInterp1D(x :: Array, y :: Array, z :: Array)
	nz	= length(z)
	nx	= length(x)
	intvals = zeros(nz,1)
	for i in 1 : nz
		zi  = z[i]
		lw	= searchsortedlast(x,zi)
		if lw == 1
			intvals[i] = y[1]
		elseif lw == nx
			intvals[i] = y[nx]
		else 
			xl  = x[lw]
			xh  = x[lw+1]
			slp = (y[lw+1] - y[lw])/(xh - xl)
			intvals[i] = y[lw] + slp*(zi - xl)
		end
	end
	return intvals
end

function LinearInterp1D(x :: Array, y :: Array, z :: Real)
	nz	= length(z)
	nx	= length(x)
	intvals = 0
	for i in 1 : nz
		zi  = z[i]
		lw	= searchsortedlast(x,zi)
		if lw == 1
			intvals = y[1]
		elseif lw == nx
			intvals = y[nx]
		else 
			xl  = x[lw]
			xh  = x[lw+1]
			slp = (y[lw+1] - y[lw])/(xh - xl)
			intvals = y[lw] + slp*(zi - xl)
		end
	end
	return intvals
end